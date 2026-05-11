import math
import logging
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

from .model_definitions import TaskType, TreeData, UnifiedModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & types
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedConfig:
    """Generator settings.

    dfs_layout: store nodes in DFS pre-order so left child is always node+1
                (no left pointer needed); right child stored as a small offset.
    quantize / quantize_bits: replace float threshold array with fixed-point
                integers and two global scalar constants.
    """
    threshold_precision: int = 4
    leaf_precision: int = 4
    pack_structs: bool = True
    quantize: bool = False
    quantize_bits: int = 16
    dfs_layout: bool = False


class DataTypeInfo(NamedTuple):
    c_type: str
    size_bytes: int
    alignment: int


class MemoryLayout(NamedTuple):
    total_bytes: int
    node_bytes: int
    threshold_bytes: int
    leaf_bytes: int
    metadata_bytes: int


class OptimizationMetrics(NamedTuple):
    original_memory: MemoryLayout
    optimized_memory: MemoryLayout
    compression_ratio: float
    threshold_deduplication: Tuple[int, int]   # (raw, unique)
    leaf_deduplication: Tuple[int, int]        # (raw, unique)
    node_size_reduction: Tuple[int, int]       # (baseline=24, optimized)


_TYPES: Dict[str, DataTypeInfo] = {
    'uint8_t':  DataTypeInfo('uint8_t',  1, 1),
    'uint16_t': DataTypeInfo('uint16_t', 2, 2),
    'uint32_t': DataTypeInfo('uint32_t', 4, 4),
}

_NATURAL_ALIGN = 4   # for unpacked structs


def _pick_type(max_val: int) -> DataTypeInfo:
    if max_val <= 255:   return _TYPES['uint8_t']
    if max_val <= 65535: return _TYPES['uint16_t']
    return _TYPES['uint32_t']


def _align(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MinimalEmbeddedTreeGenerator:
    """Parse a UnifiedModel and emit an optimized self-contained C header."""

    def __init__(self, model: UnifiedModel, config: EmbeddedConfig):
        self.model  = model
        self.config = config
        # set by analyze_and_optimize
        self.unique_thresholds: List[float] = []
        self.unique_leaves:     List[float] = []
        self.threshold_map:     Dict[float, int] = {}
        self.leaf_map:          Dict[float, int] = {}
        self.threshold_min:  float = 0.0
        self.threshold_step: float = 0.0
        self.feature_type:       Optional[DataTypeInfo] = None
        self.threshold_idx_type: Optional[DataTypeInfo] = None
        self.node_idx_type:      Optional[DataTypeInfo] = None   # BFS
        self.right_offset_type:  Optional[DataTypeInfo] = None   # DFS
        self.leaf_idx_type:      Optional[DataTypeInfo] = None
        self.node_start_indices: List[int] = []
        self.dfs_sequences:      List[List[dict]] = []           # DFS
        self.metrics: Optional[OptimizationMetrics] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_and_optimize(self) -> OptimizationMetrics:
        orig = self._original_memory()
        self._extract_unique_values()
        self._select_types()
        self._calc_node_indices()
        if self.config.dfs_layout:
            self._build_dfs_sequences()
        opt = self._optimized_memory()

        thr_orig = sum(
            sum(1 for j in range(t.node_count) if t.children_left[j] != -1)
            for t in self.model.trees
        )
        lv_orig = sum(
            sum(1 for j in range(t.node_count) if t.children_left[j] == -1)
            for t in self.model.trees
        )
        self.metrics = OptimizationMetrics(
            original_memory=orig,
            optimized_memory=opt,
            compression_ratio=orig.total_bytes / max(opt.total_bytes, 1),
            threshold_deduplication=(thr_orig, len(self.unique_thresholds)),
            leaf_deduplication=(lv_orig, len(self.unique_leaves)),
            node_size_reduction=(24, self._node_size()),
        )
        return self.metrics

    def generate_code(self, output_path: str) -> str:
        if self.metrics is None:
            raise RuntimeError("Call analyze_and_optimize() first.")
        code = self._emit_c()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        return code

    def detailed_diagnostics(self) -> str:
        if self.metrics is None:
            raise RuntimeError("Call analyze_and_optimize() first.")
        m = self.metrics
        ns = m.node_size_reduction[1]
        total_nodes = sum(max(t.node_count, 1) for t in self.model.trees)
        n_trees = len(self.model.trees)
        thr_orig, thr_uniq = m.threshold_deduplication
        lv_orig,  lv_uniq  = m.leaf_deduplication

        dfs = self.config.dfs_layout
        child_type = self.right_offset_type if dfs else self.node_idx_type
        nchild     = 1 if dfs else 2
        internal   = (self.feature_type.size_bytes
                      + self.threshold_idx_type.size_bytes
                      + nchild * child_type.size_bytes)
        union_sz   = max(internal, self.leaf_idx_type.size_bytes)

        thr_lbl = (f"uint{self.config.quantize_bits}_t (q)"
                   if self.config.quantize else self.threshold_idx_type.c_type)
        child_lbl = "right_offset" if dfs else "left / right"
        mode_lbl  = ("DFS pre-order (left=node+1, implicit)"
                     if dfs else "BFS (left+right stored)")

        W = "=" * 56
        D = "-" * 56
        lines = [
            W, " BONSAI MEMORY DIAGNOSTIC REPORT", W, "",
            "[Node Struct]",
            f"  Layout  : {mode_lbl}",
            f"  Packing : {'#pragma pack(1)' if self.config.pack_structs else 'natural align'}",
            f"  is_leaf       uint8_t  (7 bits wasted)        = 1 B",
            f"  +-- union:",
            f"  |   feature        {self.feature_type.c_type:<12}               = {self.feature_type.size_bytes} B",
            f"  |   threshold      {thr_lbl:<12}               = {self.threshold_idx_type.size_bytes} B",
            f"  |   {child_lbl:<14} {child_type.c_type:<12}               = {nchild * child_type.size_bytes} B",
            f"  |   internal struct                            = {internal} B",
            f"  |   leaf_idx       {self.leaf_idx_type.c_type:<12}               = {self.leaf_idx_type.size_bytes} B",
            f"  +-- union = max({internal},{self.leaf_idx_type.size_bytes})                  = {union_sz} B",
            f"  {D}",
            f"  sizeof(OptNode) = {ns} B  |  waste: 7 bits + {union_sz - internal} B padding",
            "",
            "[Memory Breakdown]",
            f"  {'Component':<36} {'bytes':>6}",
            D,
            f"  Nodes      {total_nodes} x {ns} B{'':>22} {m.optimized_memory.node_bytes:>6,}",
        ]
        if self.config.quantize:
            lines.append(f"  Thresholds MIN+STEP (2 floats){'':>17} {m.optimized_memory.threshold_bytes:>6,}")
        else:
            lines.append(f"  Thresholds {thr_uniq} unique x 4 B{'':>18} {m.optimized_memory.threshold_bytes:>6,}")
        lines += [
            f"  Leaves     {lv_uniq} unique x 4 B{'':>18} {m.optimized_memory.leaf_bytes:>6,}",
            f"  Metadata   {n_trees} trees x 6 B{'':>19} {m.optimized_memory.metadata_bytes:>6,}",
            D,
            f"  TOTAL FLASH{'':>31} {m.optimized_memory.total_bytes:>6,} B",
            "",
            "[vs Baseline  (naive 24 B/node struct)]",
            f"  Baseline  {m.original_memory.total_bytes:>8,} B",
            f"  Optimized {m.optimized_memory.total_bytes:>8,} B",
            f"  Saved     {m.original_memory.total_bytes - m.optimized_memory.total_bytes:>8,} B  "
            f"({100*(m.original_memory.total_bytes - m.optimized_memory.total_bytes)/m.original_memory.total_bytes:.1f}%)",
            f"  Ratio          {m.compression_ratio:.2f}x",
        ]
        if not self.config.quantize:
            lines += [
                "",
                "[Deduplication]",
                f"  Thresholds  {thr_orig} -> {thr_uniq}  "
                f"({100*(1-thr_uniq/max(thr_orig,1)):.1f}% reduction)",
            ]
        lines += [
            f"  Leaves      {lv_orig} -> {lv_uniq}  "
            f"({100*(1-lv_uniq/max(lv_orig,1)):.1f}% reduction)",
        ]

        # info-theoretic bound
        n_feats = self.model.num_features
        topo_bits = 2 * total_nodes
        feat_bpn  = math.log2(n_feats) if n_feats > 1 else 0.0
        feat_bits = thr_orig * feat_bpn
        thr_bpn   = (float(self.config.quantize_bits) if self.config.quantize
                     else (math.log2(thr_uniq) if thr_uniq > 1 else 0.0))
        thr_bits  = thr_orig * thr_bpn
        leaf_bits = lv_uniq * 32
        theory    = math.ceil((topo_bits + feat_bits + thr_bits + leaf_bits) / 8)
        eff       = 100.0 * theory / m.optimized_memory.total_bytes
        lines += [
            "",
            "[Information-Theoretic Lower Bound]",
            f"  Topology    2N bits                           {math.ceil(topo_bits/8):>6} B",
            f"  Features    log2({n_feats})={feat_bpn:.1f} b x {thr_orig} nodes    {math.ceil(feat_bits/8):>6} B",
            f"  Thresholds  {thr_bpn:.1f} b x {thr_orig} nodes             {math.ceil(thr_bits/8):>6} B",
            f"  Leaf values {lv_uniq} x float32                  {math.ceil(leaf_bits/8):>6} B",
            D,
            f"  Minimum (bit-packed)                       {theory:>6} B",
            f"  Current                                    {m.optimized_memory.total_bytes:>6} B",
            f"  Efficiency                                  {eff:.1f}%",
            "",
            W,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _original_memory(self) -> MemoryLayout:
        total = sum(t.node_count for t in self.model.trees)
        node_b = total * 24
        meta_b = len(self.model.trees) * 8   # pointer(4) + int(4)
        return MemoryLayout(node_b + meta_b, node_b, 0, 0, meta_b)

    def _extract_unique_values(self):
        raw_thr: List[float] = []
        all_lv: set = set()
        for tree in self.model.trees:
            for i in range(tree.node_count):
                if tree.children_left[i] == -1:
                    all_lv.add(round(float(tree.values[i]), self.config.leaf_precision))
                else:
                    raw_thr.append(float(tree.thresholds[i]))
        if self.config.quantize:
            t_min = min(raw_thr, default=0.0)
            t_max = max(raw_thr, default=0.0)
            bmax  = (1 << self.config.quantize_bits) - 1
            self.threshold_min  = t_min
            self.threshold_step = (t_max - t_min) / bmax if t_max > t_min else 1.0
        else:
            deduped = sorted(set(
                round(t, self.config.threshold_precision) for t in raw_thr
            ))
            self.unique_thresholds = deduped
            self.threshold_map     = {v: i for i, v in enumerate(deduped)}
        self.unique_leaves = sorted(all_lv)
        self.leaf_map      = {v: i for i, v in enumerate(self.unique_leaves)}

    def _qt(self, t: float) -> int:
        """Quantize a threshold to fixed-point integer."""
        if self.threshold_step == 0:
            return 0
        bmax = (1 << self.config.quantize_bits) - 1
        return max(0, min(bmax, round((t - self.threshold_min) / self.threshold_step)))

    def _subtree_sizes(self, tree: TreeData) -> Dict[int, int]:
        """Compute subtree sizes for every node in one DFS pass."""
        sizes: Dict[int, int] = {}
        def _dfs(i: int):
            if tree.children_left[i] == -1:
                sizes[i] = 1
            else:
                _dfs(tree.children_left[i])
                _dfs(tree.children_right[i])
                sizes[i] = 1 + sizes[tree.children_left[i]] + sizes[tree.children_right[i]]
        if tree.node_count:
            _dfs(0)
        return sizes

    def _build_dfs_sequences(self):
        """Emit nodes in DFS pre-order; store right_offset instead of left/right."""
        self.dfs_sequences = []
        for tree in self.model.trees:
            seq: List[dict] = []
            if tree.node_count == 0:
                seq.append({'leaf': True, 'leaf_idx': 0})
            else:
                sizes = self._subtree_sizes(tree)
                self._dfs_emit(tree, 0, sizes, seq)
            self.dfs_sequences.append(seq)

    def _dfs_emit(self, tree: TreeData, i: int, sizes: Dict[int, int], seq: list):
        if tree.children_left[i] == -1:
            lv = round(float(tree.values[i]), self.config.leaf_precision)
            seq.append({'leaf': True, 'leaf_idx': self.leaf_map.get(lv, 0)})
        else:
            left   = tree.children_left[i]
            roff   = 1 + sizes[left]
            feat   = int(tree.features[i])
            thresh = (self._qt(float(tree.thresholds[i]))
                      if self.config.quantize
                      else self.threshold_map.get(
                              round(float(tree.thresholds[i]), self.config.threshold_precision), 0))
            seq.append({'leaf': False, 'feature': feat, 'threshold': thresh, 'right_offset': roff})
            self._dfs_emit(tree, left, sizes, seq)
            self._dfs_emit(tree, tree.children_right[i], sizes, seq)

    def _select_types(self):
        max_feat  = max((max(t.features) for t in self.model.trees if t.features), default=0)
        max_lv    = len(self.unique_leaves) - 1 if self.unique_leaves else 0
        self.feature_type  = _pick_type(max_feat)
        self.leaf_idx_type = _pick_type(max_lv)

        if self.config.quantize:
            self.threshold_idx_type = _pick_type((1 << self.config.quantize_bits) - 1)
        else:
            self.threshold_idx_type = _pick_type(
                len(self.unique_thresholds) - 1 if self.unique_thresholds else 0
            )

        if self.config.dfs_layout:
            max_roff = 0
            for tree in self.model.trees:
                if tree.node_count:
                    sizes = self._subtree_sizes(tree)
                    for j in range(tree.node_count):
                        if tree.children_left[j] != -1:
                            max_roff = max(max_roff, 1 + sizes[tree.children_left[j]])
            self.right_offset_type = _pick_type(max_roff)
            self.node_idx_type     = None
        else:
            max_idx = 0
            for tree in self.model.trees:
                if tree.node_count:
                    mx = max(max(tree.children_left, default=0),
                             max(tree.children_right, default=0))
                    max_idx = max(max_idx, mx)
            self.node_idx_type    = _pick_type(max_idx)
            self.right_offset_type = None

    def _calc_node_indices(self):
        idx = 0
        self.node_start_indices = []
        for tree in self.model.trees:
            self.node_start_indices.append(idx)
            idx += max(tree.node_count, 1)

    def _node_size(self) -> int:
        """sizeof(OptNode) with current type selection."""
        dfs      = self.config.dfs_layout
        child    = self.right_offset_type if dfs else self.node_idx_type
        nchild   = 1 if dfs else 2
        internal = (self.feature_type.size_bytes
                    + self.threshold_idx_type.size_bytes
                    + nchild * child.size_bytes)
        if self.config.pack_structs:
            return 1 + max(internal, self.leaf_idx_type.size_bytes)
        a = _NATURAL_ALIGN
        return _align(1 + max(_align(internal, a), self.leaf_idx_type.size_bytes), a)

    def _optimized_memory(self) -> MemoryLayout:
        total  = sum(max(t.node_count, 1) for t in self.model.trees)
        node_b = total * self._node_size()
        thr_b  = 8 if self.config.quantize else len(self.unique_thresholds) * 4
        lv_b   = len(self.unique_leaves) * 4
        meta_b = len(self.model.trees) * 6   # uint32 start_idx + uint16 node_count
        return MemoryLayout(node_b + thr_b + lv_b + meta_b, node_b, thr_b, lv_b, meta_b)

    # ------------------------------------------------------------------
    # C code generation
    # ------------------------------------------------------------------

    def _emit_c(self) -> str:
        m         = self.metrics
        cfg       = self.config
        model     = self.model
        ns        = m.node_size_reduction[1]
        total     = sum(max(t.node_count, 1) for t in model.trees)
        task_str  = str(model.task_type).split('.')[-1]
        thr_field = "threshold_q" if cfg.quantize else "threshold_idx"
        dfs       = cfg.dfs_layout
        mem       = m.optimized_memory

        header = (
            f"/*\n"
            f" * Bonsai — auto-generated embedded tree model\n"
            f" * Trees: {len(model.trees)}  Features: {model.num_features}"
            f"  Task: {task_str}  Base: {model.base_score:.6f}\n"
            f" * Layout : {'DFS (left implicit, right_offset stored)' if dfs else 'BFS (left+right)'}  "
            f"Threshold: {'quantize-' + str(cfg.quantize_bits) + 'b' if cfg.quantize else 'float-dedup'}\n"
            f" * Node: {ns} B  Nodes: {total}  "
            f"Memory: {m.original_memory.total_bytes:,} B -> {mem.total_bytes:,} B ({m.compression_ratio:.2f}x)\n"
            f" */\n"
            f"#ifndef BONSAI_MODEL_H\n#define BONSAI_MODEL_H\n"
        )

        includes = "\n#include <stdint.h>\n#include <stdbool.h>\n#include <math.h>\n"

        consts = self._c_constants(total, task_str)
        struct = self._c_struct(thr_field, dfs)
        data   = self._c_data()
        nodes  = self._c_nodes(thr_field, dfs, total)
        trees  = self._c_trees()
        trav   = self._c_traversal(thr_field, dfs)
        pred   = self._c_predict()
        footer = "\n#endif /* BONSAI_MODEL_H */\n"

        return "\n".join([header, includes, consts, struct, data, nodes, trees, trav, pred, footer])

    def _c_constants(self, total: int, task_str: str) -> str:
        cfg = self.config
        lines = [
            "/* constants */",
            f"#define N_TREES     {len(self.model.trees)}",
            f"#define N_FEATURES  {self.model.num_features}",
            f"#define N_CLASSES   {self.model.num_classes}",
            f"#define N_LEAVES    {len(self.unique_leaves)}",
            f"#define TOTAL_NODES {total}",
            f"#define NODE_SIZE   {self.metrics.node_size_reduction[1]}",
            f"#define TASK_TYPE   {task_str}",
            f"#define BASE_SCORE  {self.model.base_score:.6f}f",
        ]
        if cfg.quantize:
            lines += [
                f"#define THRESHOLD_MIN  {self.threshold_min:.6f}f",
                f"#define THRESHOLD_STEP {self.threshold_step:.8f}f",
            ]
        else:
            lines += [f"#define N_THRESHOLDS {len(self.unique_thresholds)}"]
        return "\n".join(lines)

    def _c_struct(self, thr_field: str, dfs: bool) -> str:
        cfg = self.config
        pack_open  = "#pragma pack(push, 1)" if cfg.pack_structs else ""
        pack_close = "#pragma pack(pop)"     if cfg.pack_structs else ""
        child_decl = (
            f"        {self.right_offset_type.c_type} right_offset;"
            if dfs else
            f"        {self.node_idx_type.c_type} left;\n        {self.node_idx_type.c_type} right;"
        )
        return (
            f"{pack_open}\n"
            f"typedef struct {{\n"
            f"    uint8_t is_leaf;\n"
            f"    union {{\n"
            f"        struct {{\n"
            f"            {self.feature_type.c_type} feature;\n"
            f"            {self.threshold_idx_type.c_type} {thr_field};\n"
            f"{child_decl}\n"
            f"        }} internal;\n"
            f"        {self.leaf_idx_type.c_type} leaf_idx;\n"
            f"    }} u;\n"
            f"}} OptNode;\n"
            f"{pack_close}\n"
            f"\ntypedef struct {{"
            f" uint32_t start_idx; uint16_t node_count; }} OptTree;"
        )

    def _c_data(self) -> str:
        cfg = self.config
        lv_data = ", ".join(
            f"{v:.{cfg.leaf_precision}f}f" for v in self.unique_leaves
        ) if self.unique_leaves else "0.0f"

        if cfg.quantize:
            thr_type = self.threshold_idx_type.c_type
            return (
                f"static inline float decode_threshold({thr_type} q) {{\n"
                f"    return THRESHOLD_MIN + q * THRESHOLD_STEP;\n}}\n"
                f"static const float SHARED_LEAVES[N_LEAVES] = {{ {lv_data} }};"
            )
        thr_data = ", ".join(
            f"{v:.{cfg.threshold_precision}f}f" for v in self.unique_thresholds
        ) if self.unique_thresholds else "0.0f"
        return (
            f"static const float SHARED_THRESHOLDS[N_THRESHOLDS] = {{ {thr_data} }};\n"
            f"static const float SHARED_LEAVES[N_LEAVES] = {{ {lv_data} }};"
        )

    def _c_nodes(self, thr_field: str, dfs: bool, total: int) -> str:
        cfg   = self.config
        lines = []
        if dfs:
            for ti, seq in enumerate(self.dfs_sequences):
                for ni, nd in enumerate(seq):
                    last = ti == len(self.dfs_sequences)-1 and ni == len(seq)-1
                    if nd['leaf']:
                        s = f"    {{1, {{.leaf_idx={nd['leaf_idx']}}}}}"
                    else:
                        s = (f"    {{0, {{.internal={{.feature={nd['feature']},"
                             f".{thr_field}={nd['threshold']},"
                             f".right_offset={nd['right_offset']}}}}}}}")
                    lines.append(s + ("" if last else ","))
        else:
            for ti, tree in enumerate(self.model.trees):
                nc = max(tree.node_count, 1)
                for ni in range(nc):
                    last = ti == len(self.model.trees)-1 and ni == nc-1
                    if tree.node_count == 0:
                        s = "    {1, {.leaf_idx=0}}"
                    elif tree.children_left[ni] == -1:
                        lv = round(float(tree.values[ni]), cfg.leaf_precision)
                        s = f"    {{1, {{.leaf_idx={self.leaf_map.get(lv,0)}}}}}"
                    else:
                        feat = int(tree.features[ni])
                        lc, rc = tree.children_left[ni], tree.children_right[ni]
                        if cfg.quantize:
                            tq = self._qt(float(tree.thresholds[ni]))
                            s = (f"    {{0, {{.internal={{.feature={feat},"
                                 f".{thr_field}={tq},.left={lc},.right={rc}}}}}}}")
                        else:
                            tv = round(float(tree.thresholds[ni]), cfg.threshold_precision)
                            tidx = self.threshold_map.get(tv, 0)
                            s = (f"    {{0, {{.internal={{.feature={feat},"
                                 f".{thr_field}={tidx},.left={lc},.right={rc}}}}}}}")
                    lines.append(s + ("" if last else ","))

        return f"static const OptNode NODES[TOTAL_NODES] = {{\n" + "\n".join(lines) + "\n};"

    def _c_trees(self) -> str:
        rows = []
        for i, tree in enumerate(self.model.trees):
            cnt = max(tree.node_count, 1)
            comma = "," if i < len(self.model.trees)-1 else ""
            rows.append(f"    {{{self.node_start_indices[i]}, {cnt}}}{comma}")
        return "static const OptTree TREES[N_TREES] = {\n" + "\n".join(rows) + "\n};"

    def _c_traversal(self, thr_field: str, dfs: bool) -> str:
        if self.config.quantize:
            thr_expr = f"decode_threshold(node->u.internal.{thr_field})"
        else:
            thr_expr = f"SHARED_THRESHOLDS[node->u.internal.{thr_field}]"

        if dfs:
            return (
                "static inline float traverse_tree(int tree_id, const float* x) {\n"
                "    const OptNode* node = &NODES[TREES[tree_id].start_idx];\n"
                "    while (!node->is_leaf) {\n"
                f"        float t = {thr_expr};\n"
                "        if (x[node->u.internal.feature] < t) node += 1;\n"
                "        else node += node->u.internal.right_offset;\n"
                "    }\n"
                "    return SHARED_LEAVES[node->u.leaf_idx];\n"
                "}"
            )
        return (
            "static inline float traverse_tree(int tree_id, const float* x) {\n"
            "    const OptTree* tree = &TREES[tree_id];\n"
            "    uint32_t idx = tree->start_idx;\n"
            "    while (idx < tree->start_idx + tree->node_count) {\n"
            "        const OptNode* node = &NODES[idx];\n"
            "        if (node->is_leaf) return SHARED_LEAVES[node->u.leaf_idx];\n"
            f"        float t = {thr_expr};\n"
            "        idx = tree->start_idx + (x[node->u.internal.feature] < t\n"
            "              ? node->u.internal.left : node->u.internal.right);\n"
            "    }\n"
            "    return 0.0f;\n"
            "}"
        )

    def _c_predict(self) -> str:
        m = self.model
        lines = []
        if m.task_type == TaskType.REGRESSION:
            lines += [
                "float predict(const float* x) {",
                f"    float s = {m.base_score:.6f}f;",
                "    for (int i = 0; i < N_TREES; i++) s += traverse_tree(i, x);",
                "    return s;",
                "}",
            ]
        elif m.task_type == TaskType.BINARY_CLASSIFICATION:
            base_logit = (math.log(m.base_score / (1.0 - m.base_score))
                          if 0 < m.base_score < 1 else 0.0)
            lines += [
                "float predict(const float* x) {",
                f"    float s = {base_logit:.6f}f;",
                "    for (int i = 0; i < N_TREES; i++) s += traverse_tree(i, x);",
                "    return 1.0f / (1.0f + expf(-s));",
                "}",
                "static inline int predict_class(const float* x) { return predict(x) >= 0.5f; }",
            ]
        else:
            lines += [
                "void predict(const float* x, float* out) {",
                f"    for (int c = 0; c < N_CLASSES; c++) out[c] = {m.base_score:.6f}f;",
                "    for (int i = 0; i < N_TREES; i++) out[i % N_CLASSES] += traverse_tree(i, x);",
                "    float mx = out[0], s = 0.0f;",
                "    for (int c = 1; c < N_CLASSES; c++) if (out[c] > mx) mx = out[c];",
                "    for (int c = 0; c < N_CLASSES; c++) { out[c] = expf(out[c]-mx); s += out[c]; }",
                "    for (int c = 0; c < N_CLASSES; c++) out[c] /= s;",
                "}",
                "static inline int predict_class(const float* x) {",
                "    float p[N_CLASSES]; predict(x, p);",
                "    int b = 0; for (int c = 1; c < N_CLASSES; c++) if (p[c]>p[b]) b=c;",
                "    return b;",
                "}",
            ]
        return "\n".join(lines)
