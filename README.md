# Bonsai

Compiles trained XGBoost / LightGBM models into a single self-contained C header.

![Project Logo](29ae4800-da05-47fb-a5db-564c3a32c3c7.png)

```bash
python -m bonsai model.json -m dfs-q16 -o model.h
```

Include the header and call `predict(x)`. No heap, no runtime, no dependencies beyond `<math.h>`. Intended for microcontrollers, FPGAs, and other memory-constrained targets.

---

## The problem

A naive C node struct for a boosted tree is 24 B:

```c
struct Node {
    float threshold, leaf_value;
    int   feature, left, right, is_leaf;
};
```

A 100-tree XGBoost ensemble at depth 6 (~11 000 nodes) occupies 264 KB — more than the flash on most mid-range MCUs. Bonsai reduces this without changing predictions.

---

## Techniques

1. **Type narrowing** — each field stored in the smallest `uintN_t` that fits its range.
2. **Threshold deduplication** — unique thresholds pooled into `SHARED_THRESHOLDS[]`; nodes store indices.
3. **Leaf deduplication** — same idea for leaf values.
4. **DFS layout** — nodes in pre-order, so the left child is `node + 1`. Only a small right-offset is stored.
5. **Fixed-point quantization** — thresholds encoded as `uint8_t` / `uint16_t` with two global scale constants, replacing the float table.
6. **Packed structs** — `#pragma pack(1)` removes padding; the leaf and internal variants share bytes via a union.

DFS traversal compiles to a pointer walk:

```c
while (!node->is_leaf) {
    float t = SHARED_THRESHOLDS[node->u.internal.threshold_idx];
    node += (x[node->u.internal.feature] < t)
          ? 1
          : node->u.internal.right_offset;
}
return SHARED_LEAVES[node->u.leaf_idx];
```

---

## Quick start

```bash
pip install -r requirements.txt

python -m bonsai path/to/model.json                          # default mode
python -m bonsai path/to/model.json -m dfs-q16 --diag        # smallest + report
python -m bonsai path/to/model.json -m q8 -o firmware/inf.h  # custom output
```

In C:

```c
#include "model.h"

float x[N_FEATURES] = { /* ... */ };
float y = predict(x);              /* regression */
int   c = predict_class(x);        /* classification */
```

---

## Modes

| Mode      | Thresholds         | Layout | Node size | Notes                  |
|-----------|--------------------|--------|-----------|------------------------|
| `default` | float dedup        | BFS    | 5 B       | Exact, largest         |
| `q16`     | 16-bit fixed-point | BFS    | 6 B       | Tiny threshold table   |
| `q8`      | 8-bit fixed-point  | BFS    | 5 B       | Slight precision loss  |
| `dfs`     | float dedup        | DFS    | 4–5 B     | Left child implicit    |
| `dfs-q16` | 16-bit fixed-point | DFS    | 5 B       | Best balance           |
| `dfs-q8`  | 8-bit fixed-point  | DFS    | 4–5 B     | Minimum memory         |

---

## Benchmark

California Housing · 100 trees · depth 6 · 8 features.

| Model                  | Nodes  | Baseline (24 B) | default | q8       | dfs-q16  | Best ratio |
|------------------------|-------:|----------------:|--------:|---------:|---------:|-----------:|
| XGB Regression         | 11 652 | 272 KB          | 81 KB   | **64 KB**| 64 KB    | 4.27×      |
| XGB Binary             |  9 120 | 214 KB          | 68 KB   | **54 KB**| 54 KB    | 3.97×      |
| XGB Multiclass (4 cls) | 40 344 | 946 KB          | 253 KB  | **207 KB**| 207 KB  | 4.58×      |
| LGB Regression         |  6 100 | 143 KB          | 45 KB   | **35 KB**| 35 KB    | 4.10×      |
| LGB Binary             |  6 022 | 141 KB          | 47 KB   | **38 KB**| 38 KB    | 3.75×      |
| LGB Multiclass (4 cls) | 24 162 | 567 KB          | 162 KB  | **132 KB**| 132 KB  | 4.31×      |

All modes match the source framework to within floating-point rounding (max error < 1e-6 for XGB, < 1e-3 for LGB).

---

## Architecture

```
model.json ──► UniversalParser ──► UnifiedModel ──► MinimalEmbeddedTreeGenerator ──► model.h
```

- `bonsai/parser.py` — auto-detects XGBoost / LightGBM JSON.
- `bonsai/model_definitions.py` — framework-agnostic flat-array tree.
- `bonsai/generator.py` — analysis, deduplication, type selection, DFS reorder, C emission.

---

## Adding a framework

1. Add a `FooParser` in `bonsai/parser.py` returning a `UnifiedModel`.
2. Register detection in `UniversalParser.parse()`.
3. Add a case to `test_e2e.py`.

---

## Testing

```bash
pytest -v          # 36 tests: parse + generate + gcc -fsyntax-only
python test_e2e.py # full e2e: correctness, benchmark, diagnostic report
```
