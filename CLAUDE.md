# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Bonsai** is a machine-learning model-to-C compiler. It ingests trained XGBoost or LightGBM models and emits a single self-contained C header for embedded inference (no heap, no external dependencies beyond `<math.h>`).

**Repo layout:**
```
Bonsai/      # Python package: parser, generator, model definitions
legacy/      # Superseded standalone scripts (lgb2c.py, xgb2c.py)
test_e2e.py  # End-to-end test: fetches real data, trains, generates C headers
gen_minimal/ # Generated C headers (git-ignored, created by test_e2e.py)
```

## Commands

Run from repo root.

```bash
# Install dependencies
pip install -r requirements.txt

# End-to-end test (fetches California Housing, trains both frameworks,
# runs all generator modes, validates predictions, writes gen_minimal/)
py test_e2e.py

# Lint / format / type-check
black Bonsai/
flake8 Bonsai/
mypy Bonsai/
```

## Generator modes (`EmbeddedConfig`)

| Field | Default | Effect |
|---|---|---|
| `threshold_precision` | 4 | decimal places when `quantize=False` |
| `leaf_precision` | 4 | decimal places for leaf values |
| `pack_structs` | True | `#pragma pack` for minimal node size |
| `quantize` | False | store thresholds as fixed-point integers |
| `quantize_bits` | 16 | bit width for quantization (8 or 16) |

**Default mode** — thresholds deduplicated into a `float SHARED_THRESHOLDS[]` array; node struct stores an index into it.

**Quantize mode** — thresholds stored inline as `uint8_t`/`uint16_t` with two global constants (`THRESHOLD_MIN`, `THRESHOLD_STEP`); a `decode_threshold(q)` inline reconstructs the float at traversal time. Eliminates the float array; trades a small amount of precision for lower memory.

## Architecture

Pipeline: **Parser → UnifiedModel → Generator → C header**

### `Bonsai/model_definitions.py`
- `TaskType`: `REGRESSION`, `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`
- `TreeData`: flat array tree (`children_left`, `children_right`, `features`, `thresholds`, `values`)
- `UnifiedModel`: framework-agnostic wrapper with predict logic

### `Bonsai/parser.py`
- `XGBoostParser.parse(path)` → `UnifiedModel`
- `LightGBMParser.parse(path)` → `UnifiedModel`
- `UniversalParser.parse(path)` — auto-detects format from JSON keys
- **Critical**: XGBoost uses strict `<` for splits (not `<=`) — do not change

### `Bonsai/generator.py`
- `MinimalEmbeddedTreeGenerator(model, config)` + `analyze_and_optimize()` + `generate_code(path)`
- Smart type selection: `uint8_t`/`uint16_t`/`uint32_t` chosen per field based on value range
- Single global `NODES[]` array (not per-tree) for cache locality
- Generated `.h` is fully self-contained — include it and call `predict(x)` or `predict_class(x)`

## Adding a new framework

1. Add a `FooParser` class in `Bonsai/parser.py` returning a `UnifiedModel`
2. Add auto-detection in `UniversalParser.parse()`
3. Add a test case in `test_e2e.py`
