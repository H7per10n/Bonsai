#!/usr/bin/env python3
"""
tree2c_compact.py - Simplified tree model to ultra-compact C converter
Pipeline: Parser → TreeModel → Compact → C Code
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ObjType(Enum):
    REG = "regression"
    BIN = "binary"
    MULTI = "multiclass"

@dataclass
class Node:
    """Simplified node representation"""
    feat: int = -1      # feature index (-1 for leaf)
    thr: float = 0.0    # threshold
    left: int = -1      # left child index
    right: int = -1     # right child index
    val: float = 0.0    # leaf value
    
    @property
    def is_leaf(self): return self.feat < 0

@dataclass
class TreeModel:
    """Intermediate representation"""
    trees: List[List[Node]]
    n_feat: int
    n_class: int
    obj: ObjType
    base: List[float]
    lr: float = 1.0

@dataclass
class CompactTree:
    """Compact tree representation"""
    nodes: List[int]     # packed feature+threshold or leaf_idx
    children: List[int]  # left/right child indices (-1 for leaves)
    is_leaf: List[bool]  # leaf flags

# ============================================================================
# PARSER - Convert XGBoost/LightGBM to TreeModel
# ============================================================================

class Parser:
    @staticmethod
    def parse(filepath: str) -> TreeModel:
        """Parse model file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Detect format
        if 'learner' in data:  # XGBoost saved model
            return Parser._xgb_saved(data)
        elif isinstance(data, list):  # XGBoost tree dump
            return Parser._xgb_dump(data)
        elif 'tree_info' in data:  # LightGBM
            return Parser._lgb(data)
        else:
            raise ValueError("Unknown format")
    
    @staticmethod
    def _xgb_saved(data: Dict) -> TreeModel:
        """Parse XGBoost saved model format"""
        learner = data['learner']
        params = learner.get('learner_model_param', {})
        
        n_feat = int(params.get('num_feature', 0))
        n_class = int(params.get('num_class', 1))
        if n_class == 0:  # Binary classification
            n_class = 1
        base = float(params.get('base_score', 0.5))
        
        # Get objective
        obj_name = learner.get('objective', {}).get('name', 'reg:squarederror')
        if 'binary' in obj_name or 'logistic' in obj_name:
            obj = ObjType.BIN
            n_class = 1
        elif 'multi' in obj_name:
            obj = ObjType.MULTI
        else:
            obj = ObjType.REG
            n_class = 1
        
        # Parse trees from gradient_booster
        gb = learner.get('gradient_booster', {})
        model = gb.get('model', {})
        trees_data = model.get('trees', [])
        
        trees = []
        for t in trees_data:
            # Handle array-based format
            if 'base_weights' in t:
                nodes = Parser._parse_xgb_array_tree(t)
            else:
                nodes = []
                Parser._parse_xgb_node(t.get('tree', t), nodes)
            
            if nodes:
                trees.append(nodes)
        
        return TreeModel(trees, n_feat, n_class, obj, [base]*n_class, 1.0)
    
    @staticmethod
    def _parse_xgb_array_tree(t: Dict) -> List[Node]:
        """Parse XGBoost array-based tree format"""
        nodes = []
        n_nodes = len(t['base_weights'])
        
        for i in range(n_nodes):
            node = Node()
            
            # Check if leaf (no children)
            left = t['left_children'][i]
            right = t['right_children'][i]
            
            if left == -1:  # Leaf node (XGBoost uses -1 for leaves)
                node.val = t['base_weights'][i]
                node.feat = -1
            else:  # Internal node
                node.feat = t['split_indices'][i]
                node.thr = t['split_conditions'][i]
                node.left = left
                node.right = right
            
            nodes.append(node)
        
        return nodes
    
    @staticmethod
    def _xgb_dump(data: List) -> TreeModel:
        """Parse XGBoost tree dump"""
        trees = []
        max_feat = -1
        
        for tree_str in data:
            if isinstance(tree_str, str):
                tree_data = json.loads(tree_str)
            else:
                tree_data = tree_str
            
            nodes = []
            Parser._parse_xgb_node(tree_data, nodes)
            trees.append(nodes)
            
            # Track max feature
            for n in nodes:
                if n.feat >= 0:
                    max_feat = max(max_feat, n.feat)
        
        return TreeModel(trees, max_feat+1, 1, ObjType.REG, [0.5], 1.0)
    
    @staticmethod
    def _parse_xgb_node(d: Dict, nodes: List[Node]) -> int:
        """Parse XGBoost node recursively (nested format)"""
        idx = len(nodes)
        node = Node()
        nodes.append(node)
        
        if 'leaf' in d:
            node.val = float(d['leaf'])
        else:
            # Parse split
            split = d.get('split', '')
            if isinstance(split, str) and split.startswith('f'):
                node.feat = int(split[1:])
            else:
                node.feat = int(d.get('split_index', -1))
            
            node.thr = float(d.get('split_condition', 0))
            
            # Parse children
            if 'children' in d and len(d['children']) >= 2:
                node.left = Parser._parse_xgb_node(d['children'][0], nodes)
                node.right = Parser._parse_xgb_node(d['children'][1], nodes)
        
        return idx
    
    @staticmethod
    def _lgb(data: Dict) -> TreeModel:
        """Parse LightGBM format"""
        n_feat = data.get('num_feature', 0)
        n_class = data.get('num_class', 1)
        
        obj_type = data.get('objective', 'regression')
        if 'binary' in str(obj_type):
            obj = ObjType.BIN
        elif 'multiclass' in str(obj_type):
            obj = ObjType.MULTI
        else:
            obj = ObjType.REG
        
        trees = []
        for info in data.get('tree_info', []):
            nodes = []
            Parser._parse_lgb_node(info.get('tree_structure', info), nodes)
            trees.append(nodes)
        
        base = [0.0] * n_class
        lr = float(data.get('learning_rate', 1.0))
        
        return TreeModel(trees, n_feat, n_class, obj, base, lr)
    
    @staticmethod
    def _parse_lgb_node(d: Dict, nodes: List[Node]) -> int:
        """Parse LightGBM node recursively"""
        idx = len(nodes)
        node = Node()
        nodes.append(node)
        
        if 'leaf_value' in d:
            node.val = float(d['leaf_value'])
        else:
            node.feat = int(d.get('split_feature', -1))
            node.thr = float(d.get('threshold', 0))
            
            if 'left_child' in d:
                node.left = Parser._parse_lgb_node(d['left_child'], nodes)
            if 'right_child' in d:
                node.right = Parser._parse_lgb_node(d['right_child'], nodes)
        
        return idx

# ============================================================================
# GENERATOR - Convert TreeModel to C code
# ============================================================================

class Generator:
    def __init__(self, model: TreeModel, precision: int = 6):
        self.model = model
        self.precision = precision  # Decimal places for leaf values
        self.leaf_pool = []
        self.leaf_map = {}
        self.compact_trees = []
        
    def generate(self, output_path: str):
        """Generate C code"""
        # Build leaf pool
        self._build_leaf_pool()
        
        # Analyze quantization params
        f_bits, t_bits, t_min, t_scale = self._analyze_params()
        
        # Convert to compact format
        for tree in self.model.trees:
            if tree:
                self.compact_trees.append(self._to_compact(tree, f_bits, t_bits, t_min, t_scale))
        
        # Write C code
        with open(output_path, 'w') as f:
            self._write_c(f, f_bits, t_bits, t_min, t_scale)
    
    def _build_leaf_pool(self):
        """Build shared leaf value pool with precision control"""
        vals = set()
        for tree in self.model.trees:
            for node in tree:
                if node.is_leaf:
                    # Round to specified precision
                    rounded = round(float(node.val), self.precision)
                    vals.add(rounded)
        
        self.leaf_pool = sorted(vals)
        self.leaf_map = {v: i for i, v in enumerate(self.leaf_pool)}
    
    def _analyze_params(self) -> Tuple[int, int, float, float]:
        """Calculate bit allocation and quantization"""
        # Find threshold range
        thrs = []
        for tree in self.model.trees:
            for node in tree:
                if not node.is_leaf:
                    thrs.append(node.thr)
        
        # Allocate bits (ensure we have enough for features)
        max_feat = max(0, self.model.n_feat - 1)
        f_bits = min(16, max(3, int(np.ceil(np.log2(max_feat + 2)))))
        t_bits = 32 - f_bits
        
        # Quantization
        if thrs:
            t_min = min(thrs)
            t_range = max(thrs) - t_min
            t_scale = ((1 << t_bits) - 1) / t_range if t_range > 0 else 1.0
        else:
            t_min, t_scale = 0.0, 1.0
        
        return f_bits, t_bits, t_min, t_scale
    
    def _to_compact(self, tree: List[Node], fb: int, tb: int, tmin: float, tscale: float) -> CompactTree:
        """Convert tree to compact format"""
        ct = CompactTree([], [], [])
        
        for node in tree:
            ct.is_leaf.append(node.is_leaf)
            
            if node.is_leaf:
                # Store leaf index
                rounded_val = round(float(node.val), self.precision)
                ct.nodes.append(self.leaf_map[rounded_val])
                ct.children.extend([-1, -1])  # No children for leaves
            else:
                # Pack feature and threshold
                thr_q = int((node.thr - tmin) * tscale) & ((1 << tb) - 1)
                packed = (node.feat << tb) | thr_q
                ct.nodes.append(packed)
                # Store children indices
                ct.children.extend([node.left, node.right])
        
        return ct
    
    def _write_c(self, f, fb: int, tb: int, tmin: float, tscale: float):
        """Write C code"""
        # Header
        f.write(f"""// Auto-generated ultra-compact tree model
// Trees: {len(self.compact_trees)}, Features: {self.model.n_feat}
// Leaf pool: {len(self.leaf_pool)} unique values

#ifndef TREE_MODEL_H
#define TREE_MODEL_H

#include <stdint.h>
#include <math.h>

#define N_TREES {len(self.compact_trees)}
#define N_FEAT {self.model.n_feat}
#define N_CLASS {self.model.n_class}
#define LR {self.model.lr:.6f}f
#define BASE {self.model.base[0]:.6f}f

#define F_BITS {fb}
#define T_BITS {tb}
#define T_MIN {tmin:.8f}f
#define T_SCALE {tscale:.8f}f

""")
        
        # Leaf pool
        f.write(f"static const float leaves[{len(self.leaf_pool)}] = {{\n    ")
        for i in range(0, len(self.leaf_pool), 8):
            chunk = self.leaf_pool[i:i+8]
            if i > 0:
                f.write(",\n    ")
            f.write(", ".join(f"{v:.{self.precision}f}f" for v in chunk))
        f.write("\n};\n\n")
        
        # Tree data
        for i, ct in enumerate(self.compact_trees):
            # Node data
            f.write(f"static const uint32_t t{i}_nodes[{len(ct.nodes)}] = {{\n    ")
            for j in range(0, len(ct.nodes), 8):
                chunk = ct.nodes[j:j+8]
                if j > 0:
                    f.write(",\n    ")
                f.write(", ".join(f"0x{d:08x}" for d in chunk))
            f.write("\n};\n")
            
            # Children indices
            f.write(f"static const int16_t t{i}_children[{len(ct.children)}] = {{\n    ")
            for j in range(0, len(ct.children), 16):
                chunk = ct.children[j:j+16]
                if j > 0:
                    f.write(",\n    ")
                f.write(", ".join(str(c) for c in chunk))
            f.write("\n};\n")
            
            # Leaf flags
            leaf_packed = self._pack_bits(ct.is_leaf)
            f.write(f"static const uint8_t t{i}_leaf[{len(leaf_packed)}] = {{\n    ")
            for j in range(0, len(leaf_packed), 16):
                chunk = leaf_packed[j:j+16]
                if j > 0:
                    f.write(",\n    ")
                f.write(", ".join(f"0x{b:02x}" for b in chunk))
            f.write("\n};\n\n")
        
        # Tree array
        f.write("typedef struct { const uint32_t* nodes; const int16_t* children; const uint8_t* leaf; int n; } Tree;\n")
        f.write("static const Tree trees[N_TREES] = {\n")
        for i, ct in enumerate(self.compact_trees):
            f.write(f"    {{t{i}_nodes, t{i}_children, t{i}_leaf, {len(ct.nodes)}}},\n")
        f.write("};\n\n")
        
        # Prediction function
        f.write("""// Tree traversal
static inline float traverse(const Tree* t, const float* x) {
    if (t->n == 0) return 0.0f;
    
    int node = 0;
    
    while (node >= 0 && node < t->n) {
        // Check if leaf
        if (t->leaf[node >> 3] & (1 << (node & 7))) {
            return leaves[t->nodes[node]];
        }
        
        // Decode internal node
        uint32_t d = t->nodes[node];
        int feat = (d >> T_BITS) & ((1 << F_BITS) - 1);
        float thr = ((d & ((1 << T_BITS) - 1)) / T_SCALE) + T_MIN;
        
        // Navigate to child
        float fval = x[feat];
        int child_idx = node * 2;
        
        // Check for NaN and decide direction
        if (isnan(fval) || fval <= thr) {
            node = t->children[child_idx];     // left child
        } else {
            node = t->children[child_idx + 1]; // right child
        }
    }
    
    return 0.0f;  // Should not reach here
}

// Predict
float predict(const float* x) {
    float sum = BASE;
    for (int i = 0; i < N_TREES; i++) {
        sum += traverse(&trees[i], x) * LR;
    }
""")
        
        # Output transformation
        if self.model.obj == ObjType.BIN:
            f.write("    return 1.0f / (1.0f + expf(-sum));\n")
        else:
            f.write("    return sum;\n")
        
        f.write("""}\n\n""")
        
        # Add multiclass support if needed
        if self.model.obj == ObjType.MULTI:
            f.write("""// Multiclass prediction
void predict_multiclass(const float* x, float* output) {
    float scores[N_CLASS];
    for (int c = 0; c < N_CLASS; c++) {
        scores[c] = BASE;
    }
    
    for (int i = 0; i < N_TREES; i++) {
        scores[i % N_CLASS] += traverse(&trees[i], x) * LR;
    }
    
    // Softmax
    float max_score = scores[0];
    for (int c = 1; c < N_CLASS; c++) {
        if (scores[c] > max_score) max_score = scores[c];
    }
    
    float sum = 0.0f;
    for (int c = 0; c < N_CLASS; c++) {
        output[c] = expf(scores[c] - max_score);
        sum += output[c];
    }
    
    for (int c = 0; c < N_CLASS; c++) {
        output[c] /= sum;
    }
}

int predict_class(const float* x) {
    float probs[N_CLASS];
    predict_multiclass(x, probs);
    
    int best = 0;
    for (int c = 1; c < N_CLASS; c++) {
        if (probs[c] > probs[best]) best = c;
    }
    return best;
}
""")
        
        f.write("\n#endif // TREE_MODEL_H\n")
    
    def _pack_bits(self, bits: List[bool]) -> List[int]:
        """Pack bits to bytes"""
        bytes_arr = []
        for i in range(0, len(bits), 8):
            byte = sum((1 << j) if i+j < len(bits) and bits[i+j] else 0 for j in range(8))
            bytes_arr.append(byte)
        return bytes_arr

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def convert(input_file: str, output_file: str, precision: int = 6):
    """Main conversion pipeline
    
    Args:
        input_file: Input JSON model file
        output_file: Output C header file
        precision: Decimal places for leaf values (default 6)
    """
    print(f"Parsing {input_file}...")
    model = Parser.parse(input_file)
    
    print(f"Model: {len(model.trees)} trees, {model.n_feat} features, {model.obj.value}")
    
    print(f"Generating {output_file} (precision: {precision} digits)...")
    gen = Generator(model, precision)
    gen.generate(output_file)
    
    # Stats
    total_nodes = sum(len(t) for t in model.trees)
    total_leaves = sum(sum(1 for n in t if n.is_leaf) for t in model.trees)
    print(f"Stats: {total_nodes} nodes ({total_leaves} leaves), {len(gen.leaf_pool)} unique values")
    
    # Memory estimate
    node_bytes = sum(len(ct.nodes) * 4 for ct in gen.compact_trees)
    child_bytes = sum(len(ct.children) * 2 for ct in gen.compact_trees)
    leaf_flag_bytes = sum((len(ct.is_leaf) + 7) // 8 for ct in gen.compact_trees)
    pool_bytes = len(gen.leaf_pool) * 4
    total_bytes = node_bytes + child_bytes + leaf_flag_bytes + pool_bytes
    
    print(f"Memory: {total_bytes/1024:.1f} KB ({total_nodes*24/total_bytes:.1f}x compression)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        convert(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    else:
        # Default for testing
        convert('wind_turbine_xgb.json', 'model.h',6)