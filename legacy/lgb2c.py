import sys
import json
import numpy as np
import statistics
from typing import Dict, List, Any, Tuple

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class LightGBMCExporter:
    """
    Converts LightGBM models to optimized C headers for embedded deployment.
    Handles both JSON and binary model formats safely.
    """

    def __init__(self, booster: Any):
        self.booster = booster
        self.model = self._load_model_safely()
        
        # Handle objective type parsing
        self.objective_config = self._parse_objective_config()
        
        # Core model parameters
        self.num_classes = self._get_num_classes()
        self.num_features = booster.num_feature()
        self.num_trees = len(self.model["tree_info"])
        self.base_score = self._get_base_score()
        self.learning_rate = self._get_learning_rate()  # Added learning rate
        self.objective = self._get_objective()
        
        # **Move this line AFTER defining self.trees**
        self.trees = [t["tree_structure"] for t in self.model["tree_info"]]
        
        # Now that self.trees exists, analyze depths & counts
        self.tree_stats = self._analyze_tree_depths()
        self.idx_type = self._determine_index_type()

    def _load_model_safely(self):
        """Handle model loading with memory safeguards"""
        try:
            raw_model = self.booster.dump_model()
        except Exception as e:
            raise ValueError(f"Failed to parse model: {str(e)}")
            
        if not isinstance(raw_model, dict):
            raise ValueError("Invalid model format - expected dictionary")
            
        return self._validate_and_patch_model(raw_model)

    def _parse_objective_config(self) -> Dict:
        """Handle different objective representations"""
        obj = self.model.get("objective", {})
        if isinstance(obj, str):
            return {"type": obj.lower()}
        return {k.lower(): v for k, v in obj.items()}

    def _validate_and_patch_model(self, raw_model: Dict) -> Dict:
        """Handle different LightGBM model formats and versions"""
        # Patch missing fields for JSON format
        if 'num_feature' not in raw_model:
            raw_model['num_feature'] = self.booster.num_feature()
            
        if 'feature_names' not in raw_model:
            raw_model['feature_names'] = [f'f{i}' for i in range(raw_model['num_feature'])]
            
        # Patch tree nodes for split_feature naming
        for tree in raw_model['tree_info']:
            self._patch_tree_nodes(tree['tree_structure'])
            
        return raw_model

    def _patch_tree_nodes(self, node: Dict):
        """Recursively patch node keys for JSON format compatibility"""
        if 'split_feature' not in node and 'split_index' in node:
            node['split_feature'] = node['split_index']
        if 'left_child' in node:
            self._patch_tree_nodes(node['left_child'])
        if 'right_child' in node:
            self._patch_tree_nodes(node['right_child'])

    def _get_num_classes(self) -> int:
        """Extract and validate number of classes"""
        num_class = self.model.get("num_class", 1)
        try:
            return max(int(num_class), 1)
        except (ValueError, TypeError):
            return 1

    def _get_base_score(self) -> List[float]:
        """Extract initial scores from model header"""
        # First check standard location
        init_score = self.model.get("header", {}).get("init_score")
        
        # Fallback to objective's initial_value if needed
        if init_score is None:
            init_score = self.objective_config.get("initial_value", [0.0])
        
        if not isinstance(init_score, list):
            return [float(init_score)]
        return [float(s) for s in init_score]

    def _format_base_score(self) -> str:
        """
        Format base_score list as a C literal.
        - Single-element list -> "0.123456f"
        - Multi-element       -> "{0.123456f, 0.654321f, ...}"
        """
        def fmt(x: float) -> str:
            if abs(x) < 1e-4 or abs(x) > 1e6:
                return f"{x:.6e}f"
            return f"{x:.6f}f"

        if len(self.base_score) == 1:
            return fmt(self.base_score[0])
        inner = ", ".join(fmt(x) for x in self.base_score)
        return "{" + inner + "}"

    def _get_learning_rate(self) -> float:  # New method
        """Extract learning rate with fallback to 1.0"""
        return float(self.model.get("learning_rate", 1.0))

    def _get_objective(self) -> str:
        """Extract and normalize objective type"""
        obj_type = self.objective_config.get("type", "regression").lower()
        if "binary" in obj_type:
            return "binary"
        if "multiclass" in obj_type:
            return "multiclass"
        return "regression"

    def _analyze_tree_depths(self):
        """
        Analyze depth and total node count for each tree.
        Returns dict with depth stats and node counts.
        """
        depths = []
        total_nodes = 0
        max_nodes_per_tree = 0
        
        def get_depth_and_count(node):
            if 'leaf_value' in node:
                return 1, 1
            left_depth, left_count = get_depth_and_count(node['left_child'])
            right_depth, right_count = get_depth_and_count(node['right_child'])
            return 1 + max(left_depth, right_depth), 1 + left_count + right_count
        
        # Now we have self.trees defined, so iterate over them:
        for tree in self.trees:
            depth, node_count = get_depth_and_count(tree)
            depths.append(depth)
            total_nodes += node_count
            if node_count > max_nodes_per_tree:
                max_nodes_per_tree = node_count
    
        return {
            'max': max(depths),
            'min': min(depths),
            'avg': statistics.mean(depths),
            'std': statistics.stdev(depths) if len(depths) > 1 else 0.0,
            'total_nodes': total_nodes,
            'depths': depths,
            'max_nodes_per_tree': max_nodes_per_tree
        }

    def _determine_index_type(self):
        max_idx = self.tree_stats["max_nodes_per_tree"]
        if max_idx < 128:
            return np.dtype(np.int8)
        elif max_idx < 32768:
            return np.dtype(np.int16)
        else:
            return np.dtype(np.int32)

    def _flatten_tree(self, root: Dict) -> List[Dict]:
        """Flatten tree with missing value handling"""
        nodes = []
        index_map = {}

        # First pass: create index mapping
        stack = [(root, -1)]
        while stack:
            node, parent_idx = stack.pop()
            node_idx = len(nodes)
            index_map[id(node)] = node_idx
            nodes.append({
                "feature": -1,
                "threshold": 0.0,
                "left": -1,
                "right": -1,
                "missing": -1,
                "value": 0.0,
                "parent": parent_idx
            })
            
            if "leaf_value" not in node:
                stack.append((node["right_child"], node_idx))
                stack.append((node["left_child"], node_idx))

        # Second pass: populate node data
        # We need a mapping from flattened index → original node dict
        idx_to_node = [None] * len(nodes)

        def assign_mapping(nd: Dict):
            i = index_map[id(nd)]
            idx_to_node[i] = nd
            if "leaf_value" not in nd:
                assign_mapping(nd["left_child"])
                assign_mapping(nd["right_child"])

        assign_mapping(root)

        for i, flat_node in enumerate(nodes):
            original = idx_to_node[i]
            if "leaf_value" in original:
                flat_node["value"] = original["leaf_value"]
                continue

            flat_node["feature"] = original["split_feature"]
            flat_node["threshold"] = original["threshold"]
            flat_node["left"] = index_map[id(original["left_child"])]
            flat_node["right"] = index_map[id(original["right_child"])]
            flat_node["missing"] = (flat_node["left"]
                                    if original.get("default_left", False)
                                    else flat_node["right"])
            # leaf_value already handled above

        return nodes

    def _write_preamble(self, f):
        """Header boilerplate and configuration"""
        f.write(f"""#ifndef LIGHTGBM_MODEL_H
#define LIGHTGBM_MODEL_H

#include <stdint.h>
#include <math.h>

// Model configuration
#define NUM_FEATURES  {self.num_features}
#define NUM_TREES     {self.num_trees}
#define NUM_CLASSES   {self.num_classes}
#define BASE_SCORE    {self._format_base_score()}
#define LEARNING_RATE {self._format_float(self.learning_rate)}  // Added learning rate

// Memory-optimized types
typedef {self.idx_type}_t node_idx_t;
#pragma pack(push, 1)
typedef struct {{
    int16_t feature;    // -1 for leaf nodes
    float threshold;
    node_idx_t left;
    node_idx_t right;
    node_idx_t missing; // Where to route missing values
    float value;
}} TreeNode;
#pragma pack(pop)

""")

    def _format_float(self, x: float) -> str:
        """Smart float formatting for C"""
        if abs(x) < 1e-4 or abs(x) > 1e6:
            return f"{x:.6e}f"
        return f"{x:.6f}f"

    def _write_tree_data(self, f):
        """
        Write out each tree as a packed array of TreeNode structs.
        Each tree gets its own `static const TreeNode treeX[] = { ... };`
        """
        for t_idx, tree_struct in enumerate(self.trees):
            flat = self._flatten_tree(tree_struct)
            f.write(f"// Tree #{t_idx}\n")
            f.write(f"static const TreeNode tree{t_idx}[{len(flat)}] = {{\n")
            for node in flat:
                feat = node["feature"]
                thresh = self._format_float(node["threshold"])
                left = node["left"]
                right = node["right"]
                missing = node["missing"]
                val = self._format_float(node["value"])
                f.write(f"    {{ {feat}, {thresh}, {left}, {right}, {missing}, {val} }},\n")
            f.write("};\n\n")

        # Now write an array of pointers to each tree:
        f.write("// Array of tree pointers\n")
        f.write(f"static const TreeNode* trees[NUM_TREES] = {{\n")
        for t_idx in range(self.num_trees):
            f.write(f"    tree{t_idx},\n")
        f.write("};\n\n")

    def _write_regression_predictor(self, f):
        f.write("""void predict(const float* features, float* output) {
    float sum = BASE_SCORE;
    
    for (int t = 0; t < NUM_TREES; ++t) {
        const TreeNode* tree = trees[t];
        node_idx_t idx = 0;
        
        while (tree[idx].feature >= 0) {
            float fval = features[tree[idx].feature];
            node_idx_t next = isnan(fval) ? tree[idx].missing :
                          (fval <= tree[idx].threshold) ? tree[idx].left : tree[idx].right;
            idx = next;
        }
        sum += tree[idx].value * LEARNING_RATE;  // Added learning rate multiplication
    }
    
    *output = sum;
}\n\n""")

    def _write_sigmoid_predictor(self, f):
        f.write("""void predict(const float* features, float* output) {
    float sum = BASE_SCORE;
    
    for (int t = 0; t < NUM_TREES; ++t) {
        const TreeNode* tree = trees[t];
        node_idx_t idx = 0;
        
        while (tree[idx].feature >= 0) {
            float fval = features[tree[idx].feature];
            node_idx_t next = isnan(fval) ? tree[idx].missing :
                          (fval <= tree[idx].threshold) ? tree[idx].left : tree[idx].right;
            idx = next;
        }
        sum += tree[idx].value * LEARNING_RATE;  // Added learning rate
    }
    
    // Apply sigmoid for binary classification
    *output = 1.0f / (1.0f + expf(-sum));
}\n\n""")

    def _write_softmax_predictor(self, f):
        f.write("""void predict(const float* features, float* output) {
    float sums[NUM_CLASSES] = {BASE_SCORE};
    
    for (int t = 0; t < NUM_TREES; ++t) {
        const TreeNode* tree = trees[t];
        node_idx_t idx = 0;
        
        while (tree[idx].feature >= 0) {
            float fval = features[tree[idx].feature];
            node_idx_t next = isnan(fval) ? tree[idx].missing :
                          (fval <= tree[idx].threshold) ? tree[idx].left : tree[idx].right;
            idx = next;
        }
        sums[t % NUM_CLASSES] += tree[idx].value * LEARNING_RATE;  // Added learning rate
    }

    // Softmax with numerical stability
    float max_val = sums[0];
    for (int c = 1; c < NUM_CLASSES; ++c) {
        if (sums[c] > max_val) max_val = sums[c];
    }
    
    float exp_sum = 0.0f;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        sums[c] = expf(sums[c] - max_val);
        exp_sum += sums[c];
    }
    
    for (int c = 0; c < NUM_CLASSES; ++c) {
        output[c] = sums[c] / exp_sum;
    }
}\n\n""")

    def _write_prediction_logic(self, f):
        """Write out the correct predict() function based on objective"""
        if self.objective == "regression":
            self._write_regression_predictor(f)
        elif self.objective == "binary":
            self._write_sigmoid_predictor(f)
        else:  # multiclass
            self._write_softmax_predictor(f)

    def _calculate_memory(self) -> float:
        """Calculate estimated memory footprint"""
        node_size = 2 + 4 + (3 * self.idx_type.itemsize) + 4  # struct size
        tree_ptr_size = 4  # 32-bit pointers
        total = (self.tree_stats["total_nodes"] * node_size +
                self.num_trees * tree_ptr_size)
        return total / 1024.0

    def export_header(self, filename: str) -> str:
        """Generate optimized C header with memory reporting"""
        with open(filename, "w") as f:
            self._write_preamble(f)
            self._write_tree_data(f)
            self._write_prediction_logic(f)
            f.write("#endif\n")
        
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


def export_lightgbm_model(model_path: str, output_path: str) -> float:
    """Public API with safe model loading"""
    if lgb is None:
        raise RuntimeError("LightGBM package required")
    
    try:
        # Load through booster for binary files
        if model_path.endswith(('.bin', '.txt')):
            booster = lgb.Booster(model_file=model_path)
        # Load JSON models through dictionary
        else:
            with open(model_path, 'r') as f:
                model_dict = json.load(f)
            booster = lgb.Booster(model_str=json.dumps(model_dict))
            
    except Exception as e:
        raise ValueError(f"Model loading failed: {str(e)}") from e
    
    exporter = LightGBMCExporter(booster)
    return exporter.export_header(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert LightGBM model to C header")
    parser.add_argument("model", help="Input LightGBM model file")
    parser.add_argument("output", help="Output header file")
    
    args = parser.parse_args()
    
    try:
        mem_usage = export_lightgbm_model(args.model, args.output)
        print(f"Successfully exported model to {args.output}")
        print(f"Estimated memory usage: {mem_usage:.2f} KB")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
