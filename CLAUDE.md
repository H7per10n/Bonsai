# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What This Is

**Bonsai** is a machine-learning model-to-C compiler. It ingests trained XGBoost or LightGBM models and emits a single self-contained C header for embedded inference (no heap, no external dependencies beyond `<math.h>`).

## Repo layout

```
bonsai/              # Python package
  __init__.py        #   public API
  model_definitions.py  #   TaskType, TreeData, UnifiedModel
  parser.py          #   XGBoostParser, LightGBMParser, UniversalParser
  generator.py       #   MinimalEmbeddedTreeGenerator + EmbeddedConfig
  __main__.py        #   CLI entry point (python -m bonsai)
legacy/              # superseded standalone scripts (reference only)
tests/               # pytest suite
  conftest.py        #   session fixtures: data loading, model training + caching
  test_generate.py   #   18 tests: parse -> validate predictions -> write generated/
  test_compile.py    #   18 tests: gcc -fsyntax-only on each generated main.c
  models/            #   (git-ignored) cached trained models
  generated/         #   (git-ignored) model.h + main.c per combination
test_e2e.py          # standalone script: all modes, comparison table, diagnostics
gen_minimal/         # (git-ignored) headers written by test_e2e.py
```

## CLI — compile a model

```bash
# Basic usage — outputs model.h next to the input file
python -m bonsai path/to/model.json

# Choose a mode, custom output, full diagnostic report
python -m bonsai path/to/model.json -m dfs-q16 -o out.h --diag

# All modes
python -m bonsai model.json -m default   # float dedup, BFS layout
python -m bonsai model.json -m q8        # 8-bit fixed-point thresholds
python -m bonsai model.json -m q16       # 16-bit fixed-point thresholds
python -m bonsai model.json -m dfs       # float dedup, DFS layout (smallest nodes)
python -m bonsai model.json -m dfs-q8
python -m bonsai model.json -m dfs-q16
```

## Other commands

```bash
# Install dependencies
pip install -r requirements.txt

# Full pytest suite (36 tests)
pytest -v

# Stages individually
pytest tests/test_generate.py -v   # parse + validate + write generated/
pytest tests/test_compile.py  -v   # gcc -fsyntax-only on each generated main.c

# Full comparison across all modes + deep diagnostic (human-readable)
python test_e2e.py

# Lint / format / type-check
black bonsai/
flake8 bonsai/
mypy bonsai/
```

## Generator modes (`EmbeddedConfig`)

| Field | Default | Effect |
|---|---|---|
| `threshold_precision` | 4 | decimal places when `quantize=False` |
| `leaf_precision` | 4 | decimal places for leaf values |
| `pack_structs` | True | `#pragma pack(1)` for minimal node size |
| `quantize` | False | store thresholds as fixed-point integers |
| `quantize_bits` | 16 | bit width for quantization (8 or 16) |
| `dfs_layout` | False | DFS node ordering: left child is always `node+1`; only right offset stored |

**Default** — thresholds deduplicated into `float SHARED_THRESHOLDS[]`; node stores an index.

**Quantize** — thresholds stored inline as `uint8_t`/`uint16_t`; two global constants (`THRESHOLD_MIN`, `THRESHOLD_STEP`) reconstruct the float. Eliminates the float array.

**DFS layout** — nodes stored in DFS pre-order so the left child is always the next node (no left pointer needed). Right child is a small relative offset (often `uint8_t` even for large models). Removes one child field; traversal becomes a simple pointer increment. Best combined with quantize for maximum compression.

**Node size accounting** — `sizeof(OptNode)` is `1 + max(internal_struct, leaf_idx)` because `is_leaf` and the union share a packed struct. The union takes `max()` of its two branches, not their sum.

## Architecture

Pipeline: **Parser → UnifiedModel → Generator → C header**

### `bonsai/model_definitions.py`
- `TaskType`: `REGRESSION`, `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`
- `TreeData`: flat array tree (`children_left`, `children_right`, `features`, `thresholds`, `values`)
- `UnifiedModel`: framework-agnostic wrapper with predict logic

### `bonsai/parser.py`
- `XGBoostParser.parse(path)` → `UnifiedModel`
- `LightGBMParser.parse(path)` → `UniversalModel`
- `UniversalParser.parse(path)` — auto-detects format from JSON keys
- **Critical**: XGBoost uses strict `<` for splits (not `<=`) — do not change

### `bonsai/generator.py`
- `MinimalEmbeddedTreeGenerator(model, config)` + `analyze_and_optimize()` + `generate_code(path)`
- `detailed_diagnostics()` — per-field struct layout, deduplication stats, information-theoretic lower bound
- Smart type selection: `uint8_t`/`uint16_t`/`uint32_t` chosen per field from value range
- Single global `NODES[]` array for cache locality
- Generated `.h` is fully self-contained — include it and call `predict(x)`

## Adding a new framework

1. Add a `FooParser` class in `bonsai/parser.py` returning a `UnifiedModel`
2. Add auto-detection in `UniversalParser.parse()`
3. Add a test case in `test_e2e.py`
