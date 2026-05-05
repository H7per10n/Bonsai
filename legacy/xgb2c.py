import json
import statistics
import xgboost as xgb
from typing import Tuple, Dict, List, Any

class XGBoostCExporter:
    """
    Enhanced XGBoost to C exporter with proper base_score handling,
    objective-aware output processing, and improved type safety.
    """

    def __init__(self, booster: xgb.Booster):
        self.booster = booster
        self.config_json = json.loads(booster.save_config())
        self.objective, self.base_score = self._extract_objective_and_base_score()
        self.num_features = self._extract_num_features()
        self.num_classes = self._extract_num_classes()
        self._validate_model_config()

        # Process tree structure
        self.trees_json = [json.loads(js) for js in booster.get_dump(dump_format="json")]
        self.num_trees = len(self.trees_json)
        self.tree_stats = self._analyze_tree_depths()

    def _extract_num_features(self) -> int:
        """Robust num_features extraction with multiple fallback paths"""
        search_paths = [
            ['learner', 'gradient_booster', 'model', 'gbtree_model_param'],
            ['learner', 'gradient_booster', 'gbtree_model_param'],
            ['learner', 'learner_model_param']
        ]
        
        for path in search_paths:
            node = self.config_json
            try:
                for key in path:
                    node = node[key]
                return int(node['num_feature'])
            except (KeyError, TypeError):
                continue
        raise ValueError("num_feature not found in model configuration")

    def _extract_num_classes(self) -> int:
        """Safe num_class extraction with validation"""
        params = self.config_json.get('learner', {}).get('learner_model_param', {})
        num_class = int(params.get('num_class', '0'))
        objective = params.get('objective', '')
        
        if 'multi:' in objective and num_class < 2:
            raise ValueError(f"Invalid num_class {num_class} for objective {objective}")
        return max(num_class, 1)  # Ensure minimum 1 for regression

    def _extract_objective_and_base_score(self) -> Tuple[str, float]:
        """Extract objective and base_score with scientific notation handling"""
        params = self.config_json.get('learner', {}).get('learner_model_param', {})
        objective = params.get('objective', 'reg:squarederror')
        
        # Handle base_score conversion with scientific notation support
        base_score_str = params.get('base_score', '0.5')
        try:
            return objective, float(base_score_str)
        except ValueError:
            return objective, float.fromhex(base_score_str) if '0x' in base_score_str else 0.5

    def _validate_model_config(self):
        """Sanity checks for model configuration"""
        if self.num_classes > 1 and 'multi:' not in self.objective:
            print(f"Warning: num_classes={self.num_classes} but objective={self.objective}")
        if 'logit' in self.objective and self.base_score != 0.5:
            print(f"Note: Using base_score={self.base_score} with logistic objective")

    def _analyze_tree_depths(self) -> Dict[str, Any]:
        depths = []
        per_tree_nodes = []
        total_nodes = 0
        max_nodes_per_tree = 0
    
        def get_depth_and_node_count(node: Dict[str, Any]) -> Tuple[int, int]:
            if 'leaf' in node:
                return 1, 1  # Depth 1, 1 node
            left_child = node['children'][0]
            right_child = node['children'][1]
            left_depth, left_node_count = get_depth_and_node_count(left_child)
            right_depth, right_node_count = get_depth_and_node_count(right_child)
            subtree_depth = 1 + max(left_depth, right_depth)
            subtree_node_count = 1 + left_node_count + right_node_count
            return subtree_depth, subtree_node_count
    
        for tree in self.trees_json:
            tree_depth, tree_node_count = get_depth_and_node_count(tree)
            depths.append(tree_depth)
            per_tree_nodes.append(tree_node_count)
            total_nodes += tree_node_count
            if tree_node_count > max_nodes_per_tree:
                max_nodes_per_tree = tree_node_count
    
        return {
        'max': max(depths),
        'min': min(depths),
        'avg': statistics.mean(depths),
        'std': statistics.stdev(depths) if len(depths) > 1 else 0.0,
        'total_nodes': total_nodes,
        'depths': depths,
        'max_nodes_per_tree': max_nodes_per_tree,
        'per_tree_nodes': per_tree_nodes
        }

    def _flatten_tree(self, root: Dict, tree_index: int) -> List[Dict]:
        """Flatten tree with proper index handling and missing value support"""
        nodes = []
        id_map = {}
        
        # First pass: create index mapping
        stack = [root]
        while stack:
            node = stack.pop()
            node_id = node['nodeid']
            id_map[node_id] = len(nodes)
            nodes.append(None)
            if 'children' in node:
                stack.extend(node['children'])

        # Second pass: populate node data
        def process_node(node):
            idx = id_map[node['nodeid']]
            if 'leaf' in node:
                nodes[idx] = {
                    'feature': -1,
                    'threshold': 0.0,
                    'left': -1,
                    'right': -1,
                    'value': float(node['leaf']),
                    'missing': -1
                }
            else:
                missing = node.get('missing', node['yes'])
                nodes[idx] = {
                    'feature': int(node['split'][1:]),  # 'f3' -> 3
                    'threshold': float(node['split_condition']),
                    'left': id_map[node['yes']],
                    'right': id_map[node['no']],
                    'value': 0.0,
                    'missing': id_map[missing]
                }
            if 'children' in node:
                for child in node['children']:
                    process_node(child)

        process_node(root)
        return nodes

    def _generate_c_type_defs(self) -> str:
        """Generate C types based on model statistics"""
        max_nodes = max(self.tree_stats['per_tree_nodes'])
        if max_nodes < 32768:
            return "typedef int16_t node_index_t;"
        return "typedef int32_t node_index_t;"

    def _format_float(self, value: float) -> str:
        """Smart float formatting for C"""
        if 1e-4 < abs(value) < 1e6:
            return f"{value:.6f}f"
        return f"{value:.6e}f"

    def _calculate_memory(self) -> float:
        """Estimate memory usage of tree structures in KB"""
        node_size = 4 + 4 + 4 + 4 + 4 + 4  # 6 fields * 4 bytes each
        total_bytes = sum(self.tree_stats['per_tree_nodes']) * node_size
        return total_bytes / 1024

    def export_header(self, filename: str) -> str:
        """Generate full C header with proper model configuration for XGBoost."""
        with open(filename, 'w') as f:
            self._write_header_preamble(f)
            self._write_tree_arrays(f)
            self._write_prediction_functions(f)
            f.write("\n#endif\n")
    
        memory_usage = self._calculate_memory()
    
        stats = (
            "Tree Stats:\n"
            f"  Total Trees        : {self.num_trees}\n"
            f"  Max Depth          : {self.tree_stats['max']}\n"
            f"  Min Depth          : {self.tree_stats['min']}\n"
            f"  Avg Depth          : {self.tree_stats['avg']:.2f}\n"
            f"  Depth Std Dev      : {self.tree_stats['std']:.2f}\n"
            f"  Total Nodes        : {self.tree_stats['total_nodes']}\n"
            f"  Max Nodes Per Tree : {self.tree_stats['max_nodes_per_tree']}\n"
            f"\nEstimated Memory Usage: {memory_usage:.2f} KB\n"
        )
        return stats

    def _write_header_preamble(self, f):
        """Write header boilerplate and metadata"""
        f.write(f"""#ifndef XGB_MODEL_H
#define XGB_MODEL_H

#include <stdint.h>
#include <math.h>

// Model configuration
#define NUM_FEATURES  {self.num_features}
#define NUM_TREES     {self.num_trees}
#define NUM_CLASSES   {self.num_classes}
#define BASE_SCORE    {self._format_float(self.base_score)}

// Tree structure
{self._generate_c_type_defs()}

typedef struct __attribute__((packed)) {{
    int16_t feature;
    float threshold;
    node_index_t left;
    node_index_t right;
    float value;
    node_index_t missing;
}} TreeNode;

""")

    def _write_tree_arrays(self, f):
        """Write tree data structures"""
        for i, tree in enumerate(self.trees_json):
            nodes = self._flatten_tree(tree, i)
            f.write(f"static const TreeNode tree{i}[] = {{\n")
            for node in nodes:
                f.write(f"    {{{node['feature']}, {self._format_float(node['threshold'])}, "
                        f"{node['left']}, {node['right']}, {self._format_float(node['value'])}, "
                        f"{node['missing']}}},\n")
            f.write("};\n\n")
        
        f.write("static const TreeNode* const trees[NUM_TREES] = {\n")
        for i in range(self.num_trees):
            f.write(f"    tree{i},\n")
        f.write("};\n\n")

    def _write_prediction_functions(self, f):
        """Write prediction functions with objective handling"""
        if self.num_classes > 1:
            self._write_multicall_predictor(f)
        else:
            if 'logistic' in self.objective:
                self._write_binary_predictor(f)
            else:
                self._write_regression_predictor(f)

    def _write_multicall_predictor(self, f):
        f.write("""void predict(const float* features, float* output) {
    float scores[NUM_CLASSES] = {BASE_SCORE};
    
    for (int t = 0; t < NUM_TREES; ++t) {
        const TreeNode* tree = trees[t];
        node_index_t idx = 0;
        
        while (tree[idx].feature >= 0) {
            float fval = features[tree[idx].feature];
            if (isnan(fval)) {
                idx = tree[idx].missing;
            } else {
                idx = (fval <= tree[idx].threshold) ? tree[idx].left : tree[idx].right;
            }
        }
        scores[t % NUM_CLASSES] += tree[idx].value;
    }
    
    // Softmax normalization
    float max_score = scores[0];
    for (int c = 1; c < NUM_CLASSES; ++c) {
        if (scores[c] > max_score) max_score = scores[c];
    }
    
    float sum_exp = 0;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        scores[c] = expf(scores[c] - max_score);
        sum_exp += scores[c];
    }
    
    for (int c = 0; c < NUM_CLASSES; ++c) {
        output[c] = scores[c] / sum_exp;
    }
}""")

    def _write_binary_predictor(self, f):
        f.write("""void predict(const float* features, float* output) {
    float score = BASE_SCORE;
    
    for (int t = 0; t < NUM_TREES; ++t) {
        const TreeNode* tree = trees[t];
        node_index_t idx = 0;
        
        while (tree[idx].feature >= 0) {
            float fval = features[tree[idx].feature];
            if (isnan(fval)) {
                idx = tree[idx].missing;
            } else {
                idx = (fval <= tree[idx].threshold) ? tree[idx].left : tree[idx].right;
            }
        }
        score += tree[idx].value;
    }
    
    // Logistic transformation
    *output = 1.0f / (1.0f + expf(-score));
}""")

    def _write_regression_predictor(self, f):
        f.write("""void predict(const float* features, float* output) {
    float score = BASE_SCORE;
    
    for (int t = 0; t < NUM_TREES; ++t) {
        const TreeNode* tree = trees[t];
        node_index_t idx = 0;
        
        while (tree[idx].feature >= 0) {
            float fval = features[tree[idx].feature];
            if (isnan(fval)) {
                idx = tree[idx].missing;
            } else {
                idx = (fval <= tree[idx].threshold) ? tree[idx].left : tree[idx].right;
            }
        }
        score += tree[idx].value;
    }
    
    *output = score;
}""")