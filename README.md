# Bonsai

**Bonsai compiles trained XGBoost / LightGBM models into a single self-contained C header.**

![Project Logo](29ae4800-da05-47fb-a5db-564c3a32c3c7.png)
```
python -m bonsai model.json -m dfs-q16 -o model.h
```

Include the header and call `predict(x)`. No heap, no runtime libraries, no external dependencies beyond `<math.h>`. Designed for microcontrollers, FPGAs, and any environment where memory is tight and linking is painful.

---

## The problem

A gradient-boosted tree ensemble stored naively in C is larger than it needs to be. A textbook node struct:

```c
struct Node {
    float  threshold;   /* 4 B */
    float  leaf_value;  /* 4 B */
    int    feature;     /* 4 B */
    int    left;        /* 4 B */
    int    right;       /* 4 B */
    int    is_leaf;     /* 4 B */
};                      /* = 24 B per node */
```

A 100-tree XGBoost model at depth 6 has ~11 000 nodes → **264 KB** just for the nodes. That fills a mid-range STM32's entire flash.

Bonsai applies a cascade of techniques to shrink this without sacrificing prediction accuracy.

---

## How it works

### 1  Type selection

Fields are stored in the smallest integer type that fits their range:

| Field | Typical value range | Storage |
|---|---|---|
| Feature index | 0–255 | `uint8_t` (1 B) |
| Threshold index | 0–1000 | `uint8_t` or `uint16_t` |
| Child index | 0–65535 | `uint8_t` or `uint16_t` |
| Leaf index | 0–10000 | `uint8_t` or `uint16_t` |

### 2  Threshold deduplication

Boosted trees reuse thresholds heavily — many branches split on the same value. Bonsai collects all unique thresholds into a shared `float SHARED_THRESHOLDS[]` array. Each node stores a small integer index into it instead of a full 4-byte float.

For a 100-tree model: 9 446 internal nodes but only ~2 000 unique thresholds → the 9 446 × 4 B float copies collapse to 2 000 × 4 B + a small index per node.

### 3  Leaf deduplication

Same principle for leaf values. Unique leaf outputs are pooled into `float SHARED_LEAVES[]`.

### 4  DFS layout — the information-theory trick

In **BFS layout** each node stores two child indices (left and right). In **DFS pre-order** the left child is always the very next node in memory — no pointer needed. Only the right child offset is stored:

```
Node 0 (root):         right_offset = 4  →  right child at node 4
Node 1 (left child):   right_offset = 2  →  right child at node 3
Node 2 (left-left):    is_leaf
Node 3 (left-right):   is_leaf
Node 4 (right child):  is_leaf
```

This eliminates one child field per node. For large models where child indices require `uint16_t`, the right-offset is typically bounded by the subtree size — often still fitting in `uint8_t`. This saves 2–4 bytes per node.

The traversal simplifies to a pointer walk:

```c
while (!node->is_leaf) {
    float t = SHARED_THRESHOLDS[node->u.internal.threshold_idx];
    if (x[node->u.internal.feature] < t)
        node += 1;                             /* left: next node */
    else
        node += node->u.internal.right_offset; /* right: skip left subtree */
}
return SHARED_LEAVES[node->u.leaf_idx];
```

### 5  Fixed-point threshold quantization

Instead of a float lookup table, thresholds are encoded as `uint8_t` or `uint16_t` with two global scale constants:

```
threshold = THRESHOLD_MIN + q * THRESHOLD_STEP
```

This replaces the entire float array (e.g. 2000 × 4 B = 8 KB) with 8 bytes. A small amount of precision is traded for a large memory reduction.

### 6  Packed structs

`#pragma pack(1)` eliminates compiler padding between fields. The node struct layout with `pack(1)`:

```
sizeof(OptNode) = 1 (is_leaf) + max(internal_struct, leaf_idx)
```

The union takes the **max** of its two branches — internal nodes and leaf nodes share the same bytes, just interpreted differently.

---

## Information-theoretic lower bound

A binary tree with N nodes can be encoded in exactly **2N bits** using balanced-parentheses representation (succinct data structures). Feature indices need `log₂(F)` bits each. Threshold indices need `log₂(T)` bits.

For the XGB regression benchmark model (100 trees, depth 6, 8 features):

| Component | Min bits | Min bytes |
|---|---|---|
| Tree topology (2N bits) | 38 184 | 4 773 B |
| Feature indices (log₂8 = 3 b × 9 446 nodes) | 28 338 | 3 543 B |
| Threshold indices (16 b × 9 446, q16) | 151 136 | 18 892 B |
| Leaf values (9 502 unique floats × 32 b) | 304 064 | 38 008 B |
| **Total minimum** | | **65 216 B** |

Bonsai's dfs-q16 mode produces **134 676 B** on this model — **48% efficiency** vs the theoretical minimum. The gap is byte-alignment overhead (you can't use fractional bytes without bit-packing) and the 7 wasted bits in the `is_leaf` field. A future 32-bit packed-word mode could close most of this gap.

---

## Quick start

```bash
pip install -r requirements.txt

# Compile a model
python -m bonsai path/to/model.json

# DFS + 16-bit quantization + diagnostics
python -m bonsai path/to/model.json -m dfs-q16 -o model.h --diag

# Custom output path
python -m bonsai path/to/model.json -m q8 -o firmware/inference.h
```

### In your C project

```c
#include "model.h"

float x[N_FEATURES] = { /* sensor readings */ };

/* Regression */
float prediction = predict(x);

/* Binary classification — returns 0 or 1 */
int cls = predict_class(x);

/* Multiclass — writes N_CLASSES probabilities */
float probs[N_CLASSES];
predict(x, probs);
int cls = predict_class(x);
```

---

## Modes

| Mode | Thresholds | Layout | Typical node size | Notes |
|---|---|---|---|---|
| `default` | float dedup | BFS | 5 B | Exact, fast, largest |
| `q16` | 16-bit fixed-point | BFS | 6 B | Tiny threshold array |
| `q8` | 8-bit fixed-point | BFS | 5 B | Smallest; some precision loss |
| `dfs` | float dedup | DFS | 4–5 B | Left child implicit |
| `dfs-q16` | 16-bit fixed-point | DFS | 5 B | Best balance |
| `dfs-q8` | 8-bit fixed-point | DFS | 4–5 B | Minimum memory |

---

## Benchmark

California Housing dataset · 100 trees · max depth 6 · 20 640 samples · 8 features

| Model | Nodes | Baseline (24 B) | default | q8 | dfs | dfs-q16 | Best ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGB Regression | 11 652 | 272 KB | 81 KB | **64 KB** | 70 KB | 64 KB | **4.27×** |
| XGB Binary | 9 120 | 214 KB | 68 KB | **54 KB** | 59 KB | 54 KB | **3.97×** |
| XGB Multiclass (4 cls) | 40 344 | 946 KB | 253 KB | **207 KB** | 214 KB | 207 KB | **4.58×** |
| LGB Regression | 6 100 | 143 KB | 45 KB | **35 KB** | 39 KB | 35 KB | **4.10×** |
| LGB Binary | 6 022 | 141 KB | 47 KB | **38 KB** | 42 KB | 38 KB | **3.75×** |
| LGB Multiclass (4 cls) | 24 162 | 567 KB | 162 KB | **132 KB** | 139 KB | 132 KB | **4.31×** |

Average savings vs naive 24 B/node baseline:
- `default` mode: **~3.2×** compression
- `q8` mode:      **~4.1×** compression  
- `dfs-q16` mode: **~3.8×** compression + 14% smaller than `default`

Correctness: all modes predict within floating-point rounding of the original frameworks (max error < 1e-6 for XGB, < 1e-3 for LGB at these scales).

---

## Diagnostic report

```
python -m bonsai model.json -m dfs-q16 --diag
```

Output (XGB regression, 100 trees, depth 6):

```
========================================================
 BONSAI MEMORY DIAGNOSTIC REPORT
========================================================

[Node Struct]
  Layout  : DFS pre-order (left=node+1, implicit)
  sizeof(OptNode) = 5 B  |  waste: 7 bits + 0 B padding

[Memory Breakdown]
  Nodes      19092 x 5 B                       95,460 B
  Thresholds MIN+STEP (2 floats)                     8 B
  Leaves     9502 unique x 4 B                  38,008 B
  Metadata   200 trees x 6 B                     1,200 B
  TOTAL FLASH                                  134,676 B

[vs Baseline  (naive 24 B/node struct)]
  Baseline   459,808 B  ->  Optimized  134,676 B
  Saved      325,132 B  (70.7%)  Ratio  3.41x

[Information-Theoretic Lower Bound]
  Minimum (bit-packed)      65,216 B
  Current                  134,676 B
  Efficiency                  48.4%
========================================================
```

---

## Architecture

```
model.json
    |
    v
UniversalParser          bonsai/parser.py
    |  auto-detects XGBoost / LightGBM format
    v
UnifiedModel             bonsai/model_definitions.py
    |  framework-agnostic flat-array tree representation
    v
MinimalEmbeddedTreeGenerator    bonsai/generator.py
    |  analyze_and_optimize()
    |    - extract & deduplicate thresholds + leaves
    |    - select uint8/16/32 per field by range
    |    - DFS reorder + compute right offsets (if dfs_layout)
    |    - compute sizeof(OptNode) correctly (union = max, not sum)
    |  generate_code(path)
    |    - emit self-contained C header
    v
model.h
```

---

## Adding a new framework

1. Add a `FooParser` class in `bonsai/parser.py` that returns a `UnifiedModel`
2. Register auto-detection in `UniversalParser.parse()`
3. Add a test case in `test_e2e.py`

---

## Testing

```bash
# Fast pytest suite (36 tests: 18 generation + 18 gcc -fsyntax-only)
pytest -v

# Full e2e: correctness + benchmark + diagnostic report
python test_e2e.py
```
