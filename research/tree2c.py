#!/usr/bin/env python3
"""
tree2c - Universal Tree Model to C Converter
Converts XGBoost, LightGBM, and scikit-learn models to optimized C code.

Elastic optimization automatically adapts to actual data ranges for maximum efficiency.
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import os
import sys

# Optional imports
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from sklearn.tree import _tree
    import pickle
    sklearn_available = True
except ImportError:
    sklearn_available = False


class ObjectiveType(Enum):
    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"


@dataclass
class DataRange:
    """Tracks min/max ranges for optimization decisions"""
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    def update(self, val: float):
        self.min_val = min(self.min_val, val)
        self.max_val = max(self.max_val, val)
    
    @property
    def range(self) -> float:
        return self.max_val - self.min_val
    
    def optimal_type(self) -> str:
        """Get optimal C integer type for this range"""
        if -128 <= self.min_val <= 127 and self.max_val <= 127:
            return "int8_t"
        elif -32768 <= self.min_val <= 32767 and self.max_val <= 32767:
            return "int16_t"
        elif 0 <= self.min_val <= 65535 and self.max_val <= 65535:
            return "uint16_t"
        return "int32_t"
    
    def quantization_bits(self, precision: float = 0.0001) -> int:
        """Calculate bits needed for quantization with given precision"""
        if self.range == 0:
            return 0
        levels = self.range / precision
        return max(1, int(np.ceil(np.log2(levels))))


@dataclass
class TreeNode:
    """Universal tree node representation"""
    feature: int = -1
    threshold: float = 0.0
    left: int = -1
    right: int = -1
    missing: int = -1
    value: float = 0.0
    
    @property
    def is_leaf(self) -> bool:
        return self.feature < 0


@dataclass
class TreeModel:
    """Universal model representation"""
    trees: List[List[TreeNode]]
    num_features: int
    num_classes: int
    objective: ObjectiveType
    base_scores: List[float]
    learning_rate: float = 1.0
    
    # Data ranges
    feature_range: DataRange = field(default_factory=DataRange)
    threshold_range: DataRange = field(default_factory=DataRange)
    value_range: DataRange = field(default_factory=DataRange)
    index_range: DataRange = field(default_factory=DataRange)
    
    def analyze_ranges(self):
        """Analyze data ranges for optimization"""
        self.feature_range.update(0)
        self.feature_range.update(self.num_features - 1)
        
        for score in self.base_scores:
            self.value_range.update(score)
        
        for tree in self.trees:
            self.index_range.update(len(tree) - 1)
            for node in tree:
                if not node.is_leaf:
                    self.threshold_range.update(node.threshold)
                else:
                    self.value_range.update(node.value)


class ModelParser:
    """Parses models from different frameworks"""
    
    @staticmethod
    def from_file(filepath: str) -> TreeModel:
        """Load model from file, auto-detecting format"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.json':
            with open(filepath, 'r') as f:
                return ModelParser.from_json(json.load(f))
        elif ext in ['.xgb', '.ubj'] and xgb:
            return ModelParser.from_xgboost(filepath)
        elif ext in ['.lgb', '.txt', '.bin'] and lgb:
            return ModelParser.from_lightgbm(filepath)
        elif ext in ['.pkl', '.joblib'] and sklearn_available:
            return ModelParser.from_sklearn(filepath)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    @staticmethod
    def from_json(data: Dict) -> TreeModel:
        """Parse JSON model from any framework"""
        # Try XGBoost format
        if isinstance(data, list) or 'split' in str(data):
            return ModelParser._parse_xgboost_json(data)
        # Try LightGBM format
        elif 'tree_info' in data or 'tree_structure' in str(data):
            return ModelParser._parse_lightgbm_json(data)
        else:
            raise ValueError("Unknown JSON format")
    
    @staticmethod
    def from_xgboost(filepath: str) -> TreeModel:
        """Load XGBoost model"""
        booster = xgb.Booster()
        booster.load_model(filepath)
        
        # Get config and trees
        config = json.loads(booster.save_config())
        trees_json = [json.loads(t) for t in booster.get_dump(dump_format='json')]
        
        return ModelParser._parse_xgboost_json({
            'config': config,
            'trees': trees_json
        })
    
    @staticmethod
    def from_lightgbm(filepath: str) -> TreeModel:
        """Load LightGBM model"""
        booster = lgb.Booster(model_file=filepath)
        return ModelParser._parse_lightgbm_json(booster.dump_model())
    
    @staticmethod
    def from_sklearn(filepath: str) -> TreeModel:
        """Load scikit-learn model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        trees = []
        if hasattr(model, 'estimators_'):  # Ensemble
            for estimator in np.ravel(model.estimators_):
                trees.append(ModelParser._convert_sklearn_tree(estimator.tree_))
        elif hasattr(model, 'tree_'):  # Single tree
            trees.append(ModelParser._convert_sklearn_tree(model.tree_))
        
        # Determine objective
        if hasattr(model, 'n_classes_'):
            if model.n_classes_ == 2:
                objective = ObjectiveType.BINARY
                num_classes = 1
            else:
                objective = ObjectiveType.MULTICLASS
                num_classes = model.n_classes_
        else:
            objective = ObjectiveType.REGRESSION
            num_classes = 1
        
        return TreeModel(
            trees=trees,
            num_features=model.n_features_in_,
            num_classes=num_classes,
            objective=objective,
            base_scores=[0.0] * num_classes,
            learning_rate=getattr(model, 'learning_rate', 1.0)
        )
    
    @staticmethod
    def _parse_xgboost_json(data: Dict) -> TreeModel:
        """Parse XGBoost JSON format"""
        if isinstance(data, list):
            # Direct tree dump
            trees_data = data
            params = {}
        else:
            # Full model
            trees_data = data.get('trees', [])
            config = data.get('config', {})
            params = config.get('learner', {}).get('learner_model_param', {})
        
        # Extract parameters
        num_features = int(params.get('num_feature', 0))
        num_classes = max(int(params.get('num_class', '1')), 1)
        base_score = float(params.get('base_score', '0.5'))
        objective_str = params.get('objective', 'reg:squarederror')
        
        if 'binary' in objective_str or 'logistic' in objective_str:
            objective = ObjectiveType.BINARY
        elif 'multi:' in objective_str:
            objective = ObjectiveType.MULTICLASS
        else:
            objective = ObjectiveType.REGRESSION
        
        # Parse trees
        trees = []
        for tree_data in trees_data:
            nodes = []
            ModelParser._parse_xgb_node(tree_data, nodes)
            trees.append(nodes)
            
            # Update num_features if needed
            for node in nodes:
                if node.feature >= 0:
                    num_features = max(num_features, node.feature + 1)
        
        return TreeModel(
            trees=trees,
            num_features=num_features,
            num_classes=num_classes,
            objective=objective,
            base_scores=[base_score] * (num_classes if num_classes > 1 else 1)
        )
    
    @staticmethod
    def _parse_xgb_node(data: Dict, nodes: List[TreeNode]) -> int:
        """Parse XGBoost node recursively"""
        idx = len(nodes)
        node = TreeNode()
        nodes.append(node)
        
        if 'leaf' in data:
            node.value = float(data['leaf'])
            node.feature = -1
        else:
            # Parse split
            split = data.get('split', '')
            node.feature = int(split[1:]) if split.startswith('f') else int(data.get('split_feature', -1))
            node.threshold = float(data.get('split_condition', 0))
            
            # Parse children
            if 'children' in data:
                node.left = ModelParser._parse_xgb_node(data['children'][0], nodes)
                node.right = ModelParser._parse_xgb_node(data['children'][1], nodes)
                
                # Missing direction
                yes = data.get('yes', node.left)
                node.missing = node.left if yes == node.left else node.right
        
        return idx
    
    @staticmethod
    def _parse_lightgbm_json(data: Dict) -> TreeModel:
        """Parse LightGBM JSON format"""
        # Extract parameters
        num_features = data.get('num_feature', 0)
        num_classes = data.get('num_class', 1)
        
        obj = data.get('objective', {})
        obj_type = obj.get('type', 'regression') if isinstance(obj, dict) else str(obj)
        
        if 'binary' in obj_type:
            objective = ObjectiveType.BINARY
        elif 'multiclass' in obj_type:
            objective = ObjectiveType.MULTICLASS
        else:
            objective = ObjectiveType.REGRESSION
        
        # Base scores
        init_score = data.get('header', {}).get('init_score', [0.0])
        base_scores = [float(s) for s in (init_score if isinstance(init_score, list) else [init_score])]
        
        # Parse trees
        trees = []
        for info in data.get('tree_info', []):
            nodes = []
            ModelParser._parse_lgb_node(info.get('tree_structure', info), nodes)
            trees.append(nodes)
        
        return TreeModel(
            trees=trees,
            num_features=num_features,
            num_classes=num_classes,
            objective=objective,
            base_scores=base_scores,
            learning_rate=float(data.get('learning_rate', 1.0))
        )
    
    @staticmethod
    def _parse_lgb_node(data: Dict, nodes: List[TreeNode]) -> int:
        """Parse LightGBM node recursively"""
        idx = len(nodes)
        node = TreeNode()
        nodes.append(node)
        
        if 'leaf_value' in data:
            node.value = float(data['leaf_value'])
            node.feature = -1
        else:
            node.feature = int(data.get('split_feature', -1))
            node.threshold = float(data.get('threshold', 0))
            
            if 'left_child' in data:
                node.left = ModelParser._parse_lgb_node(data['left_child'], nodes)
            if 'right_child' in data:
                node.right = ModelParser._parse_lgb_node(data['right_child'], nodes)
            
            # Missing direction
            node.missing = node.left if data.get('default_left', False) else node.right
        
        return idx
    
    @staticmethod
    def _convert_sklearn_tree(tree) -> List[TreeNode]:
        """Convert sklearn tree to universal format"""
        nodes = []
        for i in range(tree.node_count):
            node = TreeNode()
            if tree.children_left[i] == _tree.TREE_LEAF:
                node.value = float(tree.value[i][0][0])
                node.feature = -1
            else:
                node.feature = int(tree.feature[i])
                node.threshold = float(tree.threshold[i])
                node.left = int(tree.children_left[i])
                node.right = int(tree.children_right[i])
                node.missing = node.left  # sklearn default
            nodes.append(node)
        return nodes


class CodeGenerator:
    """Generates optimized C code based on data ranges"""
    
    def __init__(self, model: TreeModel, optimize: bool = True):
        self.model = model
        self.optimize = optimize
        model.analyze_ranges()
        
        # Determine optimal types
        self.feature_type = model.feature_range.optimal_type() if optimize else "int32_t"
        self.index_type = model.index_range.optimal_type() if optimize else "int32_t"
        
        # Check quantization feasibility
        self.quantize_thresholds = (optimize and 
                                   model.threshold_range.quantization_bits() <= 16)
        self.quantize_values = (optimize and 
                              model.value_range.quantization_bits() <= 16)
    
    def generate(self, output_path: str) -> str:
        """Generate C header file"""
        with open(output_path, 'w') as f:
            self._write_header(f)
            self._write_config(f)
            self._write_node_struct(f)
            self._write_tree_data(f)
            self._write_predictors(f)
            f.write("\n#endif // TREE_MODEL_H\n")
        
        return self._get_stats()
    
    def _write_header(self, f):
        """Write file header"""
        f.write(f"""/*
 * Auto-generated by tree2c
 * Model: {self.model.objective.value} with {len(self.model.trees)} trees
 * Optimization: {'ENABLED' if self.optimize else 'DISABLED'}
 */

#ifndef TREE_MODEL_H
#define TREE_MODEL_H

#include <stdint.h>
#include <math.h>

""")
    
    def _write_config(self, f):
        """Write configuration macros"""
        base_scores = "{" + ", ".join(f"{s:.6f}f" for s in self.model.base_scores) + "}"
        if len(self.model.base_scores) == 1:
            base_scores = f"{self.model.base_scores[0]:.6f}f"
        
        f.write(f"""// Model configuration
#define NUM_FEATURES {self.model.num_features}
#define NUM_TREES {len(self.model.trees)}
#define NUM_CLASSES {self.model.num_classes}
#define BASE_SCORE {base_scores}
#define LEARNING_RATE {self.model.learning_rate:.6f}f

""")
        
        if self.quantize_thresholds:
            scale = 65535.0 / self.model.threshold_range.range
            f.write(f"""// Threshold quantization
#define THRESHOLD_MIN {self.model.threshold_range.min_val:.6f}f
#define THRESHOLD_SCALE {scale:.6f}f

""")
        
        if self.quantize_values:
            scale = 32767.0 / self.model.value_range.range
            f.write(f"""// Value quantization
#define VALUE_MIN {self.model.value_range.min_val:.6f}f
#define VALUE_SCALE {scale:.6f}f

""")
    
    def _write_node_struct(self, f):
        """Write node structure"""
        f.write("typedef struct __attribute__((packed)) {\n")
        f.write(f"    {self.feature_type} feature;\n")
        
        if self.quantize_thresholds:
            f.write("    uint16_t threshold_q;\n")
        else:
            f.write("    float threshold;\n")
        
        f.write(f"    {self.index_type} left;\n")
        f.write(f"    {self.index_type} right;\n")
        f.write(f"    {self.index_type} missing;\n")
        
        if self.quantize_values:
            f.write("    int16_t value_q;\n")
        else:
            f.write("    float value;\n")
        
        f.write("} TreeNode;\n\n")
        
        # Dequantization helpers
        if self.quantize_thresholds:
            f.write("""static inline float get_threshold(const TreeNode* node) {
    return node->threshold_q / THRESHOLD_SCALE + THRESHOLD_MIN;
}

""")
        
        if self.quantize_values:
            f.write("""static inline float get_value(const TreeNode* node) {
    return node->value_q / VALUE_SCALE + VALUE_MIN;
}

""")
    
    def _write_tree_data(self, f):
        """Write tree arrays"""
        for i, tree in enumerate(self.model.trees):
            f.write(f"static const TreeNode tree_{i}[] = {{\n")
            
            for node in tree:
                # Quantize if needed
                if self.quantize_thresholds and not node.is_leaf:
                    thresh = int((node.threshold - self.model.threshold_range.min_val) * 
                               (65535.0 / self.model.threshold_range.range))
                else:
                    thresh = f"{node.threshold:.6f}f"
                
                if self.quantize_values and node.is_leaf:
                    val = int((node.value - self.model.value_range.min_val) * 
                            (32767.0 / self.model.value_range.range))
                else:
                    val = f"{node.value:.6f}f"
                
                f.write(f"    {{{node.feature}, {thresh}, {node.left}, "
                       f"{node.right}, {node.missing}, {val}}},\n")
            
            f.write("};\n\n")
        
        # Tree pointer array
        f.write("static const TreeNode* const trees[] = {\n")
        for i in range(len(self.model.trees)):
            f.write(f"    tree_{i},\n")
        f.write("};\n\n")
    
    def _write_predictors(self, f):
        """Write prediction functions"""
        # Tree traversal
        f.write("""// Tree traversal
static inline float traverse_tree(const TreeNode* tree, const float* features) {
    int idx = 0;
    while (tree[idx].feature >= 0) {
        float fval = features[tree[idx].feature];
        if (__builtin_expect(isnan(fval), 0)) {
            idx = tree[idx].missing;
        } else {
""")
        
        if self.quantize_thresholds:
            f.write("            idx = (fval <= get_threshold(&tree[idx])) ? ")
        else:
            f.write("            idx = (fval <= tree[idx].threshold) ? ")
        
        f.write("tree[idx].left : tree[idx].right;\n        }\n    }\n")
        
        if self.quantize_values:
            f.write("    return get_value(&tree[idx]);\n")
        else:
            f.write("    return tree[idx].value;\n")
        
        f.write("}\n\n")
        
        # Prediction functions
        if self.model.objective == ObjectiveType.REGRESSION:
            f.write("""void predict(const float* features, float* output) {
    float sum = BASE_SCORE;
    for (int t = 0; t < NUM_TREES; ++t) {
        sum += traverse_tree(trees[t], features) * LEARNING_RATE;
    }
    *output = sum;
}
""")
        elif self.model.objective == ObjectiveType.BINARY:
            f.write("""void predict(const float* features, float* output) {
    float sum = BASE_SCORE;
    for (int t = 0; t < NUM_TREES; ++t) {
        sum += traverse_tree(trees[t], features) * LEARNING_RATE;
    }
    *output = 1.0f / (1.0f + expf(-sum));
}

int predict_class(const float* features) {
    float prob;
    predict(features, &prob);
    return prob > 0.5f ? 1 : 0;
}
""")
        else:  # MULTICLASS
            f.write("""void predict(const float* features, float* output) {
    float scores[NUM_CLASSES] = BASE_SCORE;
    
    for (int t = 0; t < NUM_TREES; ++t) {
        scores[t % NUM_CLASSES] += traverse_tree(trees[t], features) * LEARNING_RATE;
    }
    
    // Softmax
    float max_score = scores[0];
    for (int c = 1; c < NUM_CLASSES; ++c) {
        if (scores[c] > max_score) max_score = scores[c];
    }
    
    float sum = 0.0f;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        output[c] = expf(scores[c] - max_score);
        sum += output[c];
    }
    
    for (int c = 0; c < NUM_CLASSES; ++c) {
        output[c] /= sum;
    }
}

int predict_class(const float* features) {
    float probs[NUM_CLASSES];
    predict(features, probs);
    
    int best = 0;
    for (int c = 1; c < NUM_CLASSES; ++c) {
        if (probs[c] > probs[best]) best = c;
    }
    return best;
}
""")
    
    def _get_stats(self) -> str:
        """Get conversion statistics"""
        total_nodes = sum(len(tree) for tree in self.model.trees)
        
        # Calculate node size
        type_sizes = {"int8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4}
        node_size = (type_sizes[self.feature_type] + 
                    (2 if self.quantize_thresholds else 4) +
                    type_sizes[self.index_type] * 3 +
                    (2 if self.quantize_values else 4))
        
        total_bytes = total_nodes * node_size
        baseline_bytes = total_nodes * 24  # Unoptimized size
        
        return f"""
Model Statistics:
  Trees: {len(self.model.trees)}, Nodes: {total_nodes}
  Type: {self.model.objective.value}, Features: {self.model.num_features}
  
Optimization Results:
  Feature type: {self.feature_type} (saves {4 - type_sizes[self.feature_type]} bytes/node)
  Index type: {self.index_type} (saves {4 - type_sizes[self.index_type]} bytes/node)
  Quantized thresholds: {self.quantize_thresholds}
  Quantized values: {self.quantize_values}
  
Memory Usage:
  Per node: {node_size} bytes
  Total: {total_bytes / 1024:.2f} KB
  Savings: {(1 - total_bytes/baseline_bytes) * 100:.1f}%
"""


def main():
    parser = argparse.ArgumentParser(
        description="tree2c - Convert tree models to optimized C code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  - XGBoost: .xgb, .ubj, .json
  - LightGBM: .lgb, .txt, .json  
  - scikit-learn: .pkl, .joblib
        """
    )
    
    parser.add_argument("input", help="Input model file")
    parser.add_argument("output", help="Output C header file")
    parser.add_argument("--no-opt", dest="optimize", action="store_false",
                        help="Disable optimization")
    
    args = parser.parse_args()
    
    try:
        # Load and convert
        print(f"Loading {args.input}...")
        model = ModelParser.from_file(args.input)
        
        print(f"Generating {'optimized' if args.optimize else 'unoptimized'} C code...")
        generator = CodeGenerator(model, args.optimize)
        stats = generator.generate(args.output)
        
        print(f"Successfully generated {args.output}")
        print(stats)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()