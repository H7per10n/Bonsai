
import numpy as np
from model_definitions import UnifiedModel, TreeData, TaskType
from typing import List, Dict, Tuple, NamedTuple
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmbeddedConfig:
    """Simplified configuration with separate precision controls"""
    threshold_precision: int = 4
    leaf_precision: int = 4
    pack_structs: bool = True

class ArchitectureInfo:
    """Simplified architecture info (assume 32-bit generic)"""
    pointer_size: int = 4
    alignment: int = 4
    bool_size: int = 1
    pack_support: bool = True

class MemoryLayout(NamedTuple):
    """Memory layout analysis"""
    total_bytes: int
    node_bytes: int
    threshold_bytes: int
    leaf_bytes: int
    metadata_bytes: int

class OptimizationMetrics(NamedTuple):
    """Optimization results"""
    original_memory: MemoryLayout
    optimized_memory: MemoryLayout
    compression_ratio: float
    threshold_deduplication: Tuple[int, int]
    leaf_deduplication: Tuple[int, int]
    node_size_reduction: Tuple[int, int]

class DataTypeInfo(NamedTuple):
    """Information about C data types"""
    c_type: str
    size_bytes: int
    alignment: int
    min_val: int
    max_val: int

class OptimizedTreeNode:
    """Optimized tree node with memory-aware design"""
    
    def __init__(self, is_leaf: bool, pack_structs: bool = False):
        self.is_leaf = is_leaf
        self.pack_structs = pack_structs
        self.arch_info = ArchitectureInfo()
        
    def calculate_size(self, feature_type: DataTypeInfo, threshold_idx_type: DataTypeInfo, node_idx_type: DataTypeInfo, leaf_idx_type: DataTypeInfo) -> int:
        """Calculate actual struct size with proper alignment"""
        if self.pack_structs:
            return 1 + feature_type.size_bytes + threshold_idx_type.size_bytes + (2 * node_idx_type.size_bytes) + leaf_idx_type.size_bytes
        else:
            bool_size = 1  # Bitfield uses 1 byte
            internal_size = feature_type.size_bytes + threshold_idx_type.size_bytes + (2 * node_idx_type.size_bytes)
            internal_aligned = self._align_size(internal_size, self.arch_info.alignment)
            leaf_aligned = self._align_size(leaf_idx_type.size_bytes, leaf_idx_type.alignment)
            union_size = max(internal_aligned, leaf_aligned)
            total_size = bool_size + union_size
            return self._align_size(total_size, self.arch_info.alignment)
    
    def _align_size(self, size: int, alignment: int) -> int:
        return ((size + alignment - 1) // alignment) * alignment

class MinimalEmbeddedTreeGenerator:
    """Minimal memory embedded tree generator with deduplication and smart types"""
    
    def __init__(self, model: UnifiedModel, config: EmbeddedConfig):
        self.model = model
        self.config = config
        self.arch_info = ArchitectureInfo()
        self.type_info = self._initialize_type_info()
        self.unique_thresholds: List[float] = []
        self.unique_leaves: List[float] = []
        self.threshold_map: Dict[float, int] = {}
        self.leaf_map: Dict[float, int] = {}
        self.feature_type: DataTypeInfo = None
        self.threshold_idx_type: DataTypeInfo = None
        self.node_idx_type: DataTypeInfo = None
        self.leaf_idx_type: DataTypeInfo = None
        self.metrics: OptimizationMetrics = None
        self.node_start_indices: List[int] = []
        
    def _initialize_type_info(self) -> Dict[str, DataTypeInfo]:
        """Initialize C data type information"""
        return {
            'uint8_t': DataTypeInfo('uint8_t', 1, 1, 0, 255),
            'uint16_t': DataTypeInfo('uint16_t', 2, 2, 0, 65535),
            'uint32_t': DataTypeInfo('uint32_t', 4, 4, 0, 4294967295),
        }
    
    def _select_optimal_type(self, max_value: int) -> DataTypeInfo:
        """Select optimal unsigned integer type for given maximum value"""
        if max_value <= 255:
            return self.type_info['uint8_t']
        elif max_value <= 65535:
            return self.type_info['uint16_t']
        else:
            return self.type_info['uint32_t']
    
    def _calculate_original_memory(self) -> MemoryLayout:
        """Calculate original memory usage with proper struct alignment"""
        total_nodes = sum(tree.node_count for tree in self.model.trees)
        original_node_size = 24 
        node_bytes = total_nodes * original_node_size
        tree_metadata = len(self.model.trees) * (self.arch_info.pointer_size + 4)
        
        return MemoryLayout(
            total_bytes=node_bytes + tree_metadata,
            node_bytes=node_bytes,
            threshold_bytes=0,
            leaf_bytes=0,
            metadata_bytes=tree_metadata
        )
    
    def analyze_and_optimize(self) -> OptimizationMetrics:
        """Perform analysis and optimization"""
        logger.info("Starting minimal memory optimization")
        
        original_memory = self._calculate_original_memory()
        self._extract_unique_values()
        self._select_optimal_types()
        self._calculate_node_indices()
        optimized_memory = self._calculate_optimized_memory()
        
        self.metrics = OptimizationMetrics(
            original_memory=original_memory,
            optimized_memory=optimized_memory,
            compression_ratio=original_memory.total_bytes / max(optimized_memory.total_bytes, 1),
            threshold_deduplication=self._calculate_threshold_deduplication(),
            leaf_deduplication=self._calculate_leaf_deduplication(),
            node_size_reduction=(24 , 
                               self._calculate_optimized_node_size())
        )
        
        logger.info(f"Optimization complete. Compression ratio: {self.metrics.compression_ratio:.2f}x")
        return self.metrics
    
    def _extract_unique_values(self):
        """Extract globally unique thresholds and leaf values with separate precision"""
        all_thresholds = set()
        all_leaves = set()
        
        for tree in self.model.trees:
            for i in range(tree.node_count):
                if tree.children_left[i] == -1:
                    leaf_val = round(float(tree.values[i]), self.config.leaf_precision)
                    all_leaves.add(leaf_val)
                else:
                    threshold_val = round(float(tree.thresholds[i]), self.config.threshold_precision)
                    all_thresholds.add(threshold_val)
        
        self.unique_thresholds = sorted(all_thresholds)
        self.unique_leaves = sorted(all_leaves)
        self.threshold_map = {v: i for i, v in enumerate(self.unique_thresholds)}
        self.leaf_map = {v: i for i, v in enumerate(self.unique_leaves)}
        
        logger.info(f"Extracted {len(self.unique_thresholds)} unique thresholds, "
                   f"{len(self.unique_leaves)} unique leaves")
    
    def _select_optimal_types(self):
        """Select optimal data types based on value ranges"""
        max_feature = max((max(tree.features) for tree in self.model.trees if tree.features), default=0)
        max_threshold_idx = len(self.unique_thresholds) - 1 if self.unique_thresholds else 0
        max_leaf_idx = len(self.unique_leaves) - 1 if self.unique_leaves else 0
        max_relative_node_idx = 0
        for tree in self.model.trees:
            if tree.node_count > 0:
                tree_max_idx = max(max(tree.children_left, default=0), max(tree.children_right, default=0))
                max_relative_node_idx = max(max_relative_node_idx, tree_max_idx)
        
        self.feature_type = self._select_optimal_type(max_feature)
        self.threshold_idx_type = self._select_optimal_type(max_threshold_idx)
        self.node_idx_type = self._select_optimal_type(max_relative_node_idx)
        self.leaf_idx_type = self._select_optimal_type(max_leaf_idx)
        
        logger.info(f"Selected types: feature={self.feature_type.c_type}, "
                   f"threshold_idx={self.threshold_idx_type.c_type}, "
                   f"node_idx={self.node_idx_type.c_type}, "
                   f"leaf_idx={self.leaf_idx_type.c_type}")
    
    def _calculate_node_indices(self):
        """Calculate start indices for single node array"""
        current_idx = 0
        self.node_start_indices = []
        for tree in self.model.trees:
            self.node_start_indices.append(current_idx)
            current_idx += max(tree.node_count, 1)
    
    def _calculate_optimized_memory(self) -> MemoryLayout:
        """Calculate optimized memory usage with single node array"""
        total_nodes = sum(max(tree.node_count, 1) for tree in self.model.trees)
        node_size = self._calculate_optimized_node_size()
        node_bytes = total_nodes * node_size
        threshold_bytes = len(self.unique_thresholds) * 4
        leaf_bytes = len(self.unique_leaves) * 4
        tree_struct_size = 4 + 2
        metadata_bytes = len(self.model.trees) * tree_struct_size
        total_bytes = node_bytes + threshold_bytes + leaf_bytes + metadata_bytes
        
        return MemoryLayout(
            total_bytes=total_bytes,
            node_bytes=node_bytes,
            threshold_bytes=threshold_bytes,
            leaf_bytes=leaf_bytes,
            metadata_bytes=metadata_bytes
        )
    
    def _calculate_optimized_node_size(self) -> int:
        """Calculate optimized node structure size"""
        dummy_node = OptimizedTreeNode(False, self.config.pack_structs)
        return dummy_node.calculate_size(self.feature_type, self.threshold_idx_type, self.node_idx_type, self.leaf_idx_type)
    
    def _calculate_threshold_deduplication(self) -> Tuple[int, int]:
        """Calculate threshold deduplication metrics"""
        original_count = sum(
            len([i for i in range(tree.node_count) if tree.children_left[i] != -1])
            for tree in self.model.trees
        )
        return (original_count, len(self.unique_thresholds))
    
    def _calculate_leaf_deduplication(self) -> Tuple[int, int]:
        """Calculate leaf deduplication metrics"""
        original_count = sum(
            len([i for i in range(tree.node_count) if tree.children_left[i] == -1])
            for tree in self.model.trees
        )
        return (original_count, len(self.unique_leaves))
    
    def generate_code(self, output_path: str) -> str:
        """Generate optimized C code matching the provided template"""
        if self.metrics is None:
            raise RuntimeError("Must call analyze_and_optimize() before generating code")
        
        code_generator = CCodeGenerator(self)
        c_code = code_generator.generate()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(c_code)
        
        logger.info(f"Generated code written to {output_path}")
        return c_code

class CCodeGenerator:
    """Professional C code generator for optimized template"""
    
    def __init__(self, generator: MinimalEmbeddedTreeGenerator):
        self.gen = generator
        self.model = generator.model
        self.config = generator.config
        self.arch_info = generator.arch_info
        self.metrics = generator.metrics
    
    def generate(self) -> str:
        """Generate complete C code matching the template"""
        total_nodes = sum(max(t.node_count, 1) for t in self.gen.model.trees)
        task_type_str = str(self.model.task_type).split('.')[-1]
        
        # Generate node data
        node_data_lines = []
        for i, tree in enumerate(self.gen.model.trees):
            node_count = max(tree.node_count, 1)
            for j in range(node_count):
                if tree.node_count == 0:
                    node_data_lines.append("    {true, {.leaf_idx=0}}")
                    continue
                    
                if tree.children_left[j] == -1:
                    leaf_val = round(float(tree.values[j]), self.config.leaf_precision)
                    leaf_idx = self.gen.leaf_map.get(leaf_val, 0)
                    node_str = f"    {{true, {{.leaf_idx={leaf_idx}}}}}"
                else:
                    feature = int(tree.features[j])
                    threshold_val = round(float(tree.thresholds[j]), self.config.threshold_precision)
                    threshold_idx = self.gen.threshold_map.get(threshold_val, 0)
                    left = tree.children_left[j]
                    right = tree.children_right[j]
                    node_str = (f"    {{false, {{.internal={{.feature={feature}, "
                              f".threshold_idx={threshold_idx}, .left={left}, .right={right}}}}}}}")
                
                if i < len(self.gen.model.trees) - 1 or j < node_count - 1:
                    node_str += ","
                node_data_lines.append(node_str)
        
        # Generate tree metadata
        tree_metadata = []
        for i, tree in enumerate(self.gen.model.trees):
            count = max(tree.node_count, 1)
            tree_metadata.append(f"    {{{self.gen.node_start_indices[i]}, {count}}}")
            if i < len(self.gen.model.trees) - 1:
                tree_metadata[-1] += ","
        
        # Generate thresholds and leaves
        threshold_data = ", ".join(f"{t:.{self.config.threshold_precision}f}f" 
                                 for t in self.gen.unique_thresholds) if self.gen.unique_thresholds else "0.0f"
        leaf_data = ", ".join(f"{l:.{self.config.leaf_precision}f}f" 
                            for l in self.gen.unique_leaves) if self.gen.unique_leaves else "0.0f"
        
        # Build template
        sections = [
            f"""/*
 * Ultra-Minimal Memory Embedded Tree Ensemble Model
 * 
 * Model Metadata:
 *   - Number of trees: {len(self.gen.model.trees)}
 *   - Number of features: {self.gen.model.num_features}
 *   - Number of classes: {self.gen.model.num_classes}
 *   - Task type: {task_type_str}
 *   - Base score: {self.model.base_score:.6f}
 * 
 * Optimization Configuration:
 *   - Threshold precision: {self.config.threshold_precision} decimal places
 *   - Leaf precision: {self.config.leaf_precision} decimal places
 *   - Packed structs: {'Yes' if self.config.pack_structs else 'No'}
 *   - Node size: {self.metrics.node_size_reduction[1]} bytes
 *   - Total nodes: {total_nodes}
 * 
 * Memory Usage:
 *   - Original: {self.metrics.original_memory.total_bytes:,} bytes
 *   - Optimized: {self.metrics.optimized_memory.total_bytes:,} bytes
 *   - Compression: {self.metrics.compression_ratio:.2f}x
 *   - Threshold deduplication: {self.metrics.threshold_deduplication[0]} -> {self.metrics.threshold_deduplication[1]}
 *   - Leaf deduplication: {self.metrics.leaf_deduplication[0]} -> {self.metrics.leaf_deduplication[1]}
 * 
 * IMPORTANT: This file is automatically generated. Do not edit manually.
 */

#ifndef EMBEDDED_TREES_MINIMAL_H
#define EMBEDDED_TREES_MINIMAL_H
""",
            """
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif
""",
            f"""
/* Model constants */
#define N_TREES {len(self.gen.model.trees)}
#define N_FEATURES {self.gen.model.num_features}
#define N_CLASSES {self.gen.model.num_classes}
#define N_THRESHOLDS {len(self.gen.unique_thresholds)}
#define N_LEAVES {len(self.gen.unique_leaves)}
#define TASK_TYPE {task_type_str}
#define BASE_SCORE {self.model.base_score:.6f}f
#define THRESHOLD_PRECISION {self.config.threshold_precision}
#define LEAF_PRECISION {self.config.leaf_precision}
#define PACK_STRUCTS {'1' if self.config.pack_structs else '0'}
#define NODE_SIZE {self.metrics.node_size_reduction[1]}
#define TOTAL_NODES {total_nodes}
#define ORIGINAL_MEMORY {self.metrics.original_memory.total_bytes}
#define OPTIMIZED_MEMORY {self.metrics.optimized_memory.total_bytes}
#define COMPRESSION_RATIO {self.metrics.compression_ratio:.2f}
#define ORIGINAL_THRESHOLDS {self.metrics.threshold_deduplication[0]}
#define UNIQUE_THRESHOLDS {self.metrics.threshold_deduplication[1]}
#define ORIGINAL_LEAVES {self.metrics.leaf_deduplication[0]}
#define UNIQUE_LEAVES {self.metrics.leaf_deduplication[1]}
""",
            f"""
/* Optimized tree node structure */
{'#pragma pack(push, 1)' if self.config.pack_structs else ''}
typedef struct {{
    uint8_t is_leaf : 1;  /* Bitfield for minimal bool size */
    union {{
        struct {{
            {self.gen.feature_type.c_type} feature;
            {self.gen.threshold_idx_type.c_type} threshold_idx;
            {self.gen.node_idx_type.c_type} left;
            {self.gen.node_idx_type.c_type} right;
        }} internal;
        {self.gen.leaf_idx_type.c_type} leaf_idx;
    }} u;
}} OptNode;
{'#pragma pack(pop)' if self.config.pack_structs else ''}

typedef struct {{
    uint32_t start_idx;  /* Start index in global node array */
    uint16_t node_count;
}} OptTree;
""",
            f"""
/* Shared data arrays for deduplication */
static const float SHARED_THRESHOLDS[N_THRESHOLDS] = {{ {threshold_data} }};
static const float SHARED_LEAVES[N_LEAVES] = {{ {leaf_data} }};
""",
            f"""
/* Single global node array */
static const OptNode NODES[TOTAL_NODES] = {{
{'\n'.join(node_data_lines)}
}};
""",
            f"""
/* Tree metadata */
static const OptTree TREES[N_TREES] = {{
{'\n'.join(tree_metadata)}
}};
""",
            """
/* Tree traversal function */
static inline float traverse_tree(int tree_id, const float* x) {
    const OptTree* tree = &TREES[tree_id];
    uint32_t node_idx = tree->start_idx;

    while (node_idx < tree->start_idx + tree->node_count) {
        const OptNode* node = &NODES[node_idx];

        if (node->is_leaf) {
            return SHARED_LEAVES[node->u.leaf_idx];
        }

        float threshold = SHARED_THRESHOLDS[node->u.internal.threshold_idx];

        if (x[node->u.internal.feature] < threshold) {
            node_idx = tree->start_idx + node->u.internal.left;
        } else {
            node_idx = tree->start_idx + node->u.internal.right;
        }
    }

    return 0.0f; /* Should never reach here */
}
""",
            self._generate_predict_function(),
            f"""
/*
// Memory breakdown:
// - Nodes: {self.metrics.optimized_memory.node_bytes} bytes ({total_nodes} x {self.metrics.node_size_reduction[1]})
// - Thresholds: {self.metrics.optimized_memory.threshold_bytes} bytes (N_THRESHOLDS x 4)
// - Leaves: {self.metrics.optimized_memory.leaf_bytes} bytes (N_LEAVES x 4)
// - Metadata: {self.metrics.optimized_memory.metadata_bytes} bytes (N_TREES x (4 + 2))
// - Total: {self.metrics.optimized_memory.total_bytes} bytes
// Alignment: {self.arch_info.alignment} bytes
// Feature type: {self.gen.feature_type.c_type}
// Threshold index type: {self.gen.threshold_idx_type.c_type}
// Node index type: {self.gen.node_idx_type.c_type}
// Leaf index type: {self.gen.leaf_idx_type.c_type}
*/
""",
            """
#ifdef __cplusplus
}
#endif

#endif /* EMBEDDED_TREES_MINIMAL_H */
"""
        ]
        
        return "\n".join(sections)
    
    def _generate_predict_function(self) -> str:
        """Generate main prediction function"""
        lines = []
        if self.model.task_type == TaskType.REGRESSION:
            lines.append("float predict(const float* x) {")
            lines.append(f"    float sum = {self.model.base_score:.6f}f;")
            lines.append("    for (int i = 0; i < N_TREES; i++) {")
            lines.append("        sum += traverse_tree(i, x);")
            lines.append("    }")
            lines.append("    return sum;")
            lines.append("}")
        elif self.model.task_type == TaskType.BINARY_CLASSIFICATION:
            base_logit = math.log(self.model.base_score / (1.0 - self.model.base_score)) if 0 < self.model.base_score < 1 else 0.0
            lines.append("float predict(const float* x) {")
            lines.append(f"    float logit = {base_logit:.6f}f;")
            lines.append("    for (int i = 0; i < N_TREES; i++) {")
            lines.append("        logit += traverse_tree(i, x);")
            lines.append("    }")
            lines.append("    return 1.0f / (1.0f + expf(-logit));")
            lines.append("}")
        else:  # Multiclass
            lines.append("void predict(const float* x, float* output) {")
            lines.append("    for (int c = 0; c < N_CLASSES; c++) {")
            lines.append(f"        output[c] = {self.model.base_score:.6f}f;")
            lines.append("    }")
            lines.append("    for (int i = 0; i < N_TREES; i++) {")
            lines.append("        output[i % N_CLASSES] += traverse_tree(i, x);")
            lines.append("    }")
            lines.append("    float max_val = output[0];")
            lines.append("    for (int c = 1; c < N_CLASSES; c++) {")
            lines.append("        if (output[c] > max_val) max_val = output[c];")
            lines.append("    }")
            lines.append("    float sum = 0.0f;")
            lines.append("    for (int c = 0; c < N_CLASSES; c++) {")
            lines.append("        output[c] = expf(output[c] - max_val);")
            lines.append("        sum += output[c];")
            lines.append("    }")
            lines.append("    for (int c = 0; c < N_CLASSES; c++) {")
            lines.append("        output[c] /= sum;")
            lines.append("    }")
            lines.append("}")
        
        return "\n".join(lines)

def optimize_model_for_embedded(
    model: UnifiedModel,
    threshold_precision: int = 4,
    leaf_precision: int = 4,
    pack_structs: bool = True,
    output_path: str = "minimal_trees.h"
) -> OptimizationMetrics:
    """
    High-level function to optimize a model for minimal memory embedded deployment
    
    Args:
        model: The tree ensemble model to optimize
        threshold_precision: Decimal precision for thresholds
        leaf_precision: Decimal precision for leaf values
        pack_structs: Use packed structs for minimal size
        output_path: Output file path for generated C code
    
    Returns:
        Optimization metrics showing memory savings
    """
    config = EmbeddedConfig(
        threshold_precision=threshold_precision,
        leaf_precision=leaf_precision,
        pack_structs=pack_structs
    )
    
    generator = MinimalEmbeddedTreeGenerator(model, config)
    metrics = generator.analyze_and_optimize()
    generator.generate_code(output_path)
    
    return metrics
