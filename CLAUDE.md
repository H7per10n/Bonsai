# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Bonsai** is a machine-learning model-to-C compiler. It ingests trained XGBoost or LightGBM models and emits a single self-contained C header suitable for embedded inference (no heap, no external deps).

**Repo layout:**
```
Bonsai/          # Python package (parser, generator, tests, saved models)
gen_minimal/     # Generated C headers + validation harnesses (6 framework×task combos)
examples/        # Jupyter notebooks and wind turbine example models
data/            # Training datasets (T1.csv, EngineFaultDB_Final.csv)
legacy/          # Superseded standalone scripts (lgb2c.py, xgb2c.py)
research/        # Experimental converters (tree2c.py, tree2c_compact.py)
embedded/        # ARM/embedded board test harnesses (verify.c, memprof.sh)
```

## Commands

Run from repo root.

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run all tests
pytest Bonsai/

# Run a single test file
pytest Bonsai/test_parser.py
pytest Bonsai/test_pipeline.py

# Run with coverage
pytest --cov=Bonsai Bonsai/

# Lint / format / type-check
black Bonsai/
flake8 Bonsai/
mypy Bonsai/

# Build and validate all generated C models
cd gen_minimal && make all && make run-all
```

## Architecture

The pipeline is: **Parser → UnifiedModel → Generator → C header**.

### `Bonsai/model_definitions.py` — core data structures
- `TaskType` enum: `REGRESSION`, `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`
- `TreeData`: flat array representation of one tree (`children_left`, `children_right`, `features`, `thresholds`, `values`)
- `UnifiedModel`: framework-agnostic wrapper; holds a list of `TreeData` plus metadata (task type, base score, num_class)

### `Bonsai/parser.py` — framework loaders
- `XGBoostParser.parse(path)` → `UnifiedModel` — reads XGBoost JSON
- `LightGBMParser.parse(path)` → `UnifiedModel` — reads LightGBM JSON (recursive tree structure flattened to arrays)
- `detect_task_type(objective_string)` — infers task from XGBoost/LightGBM objective strings
- **Critical quirk**: XGBoost splits use strict `<` (not `<=`); this is empirically verified and must not change

### `Bonsai/generator.py` — C code emitter
- `MinimalEmbeddedTreeGenerator` driven by an `EmbeddedConfig` (threshold/leaf precision, type-width hints)
- Optimizations: threshold and leaf-value deduplication into shared arrays; `uint8_t`/`uint16_t`/`uint32_t` type selection based on value range; packed 6-byte node structs; single global node array (not per-tree)
- Output: one `.h` header with `#define`, struct declarations, and embedded arrays — no `malloc`, no external includes beyond `<stdint.h>`

### `gen_minimal/` — generated outputs
Six subdirectories, one per framework × task-type combination. Each contains `*.h` (generated header), `main.c` (validation harness), and `Makefile` (gcc -O2). The top-level `gen_minimal/Makefile` builds and runs all variants.

### Tests
- `Bonsai/test_parser.py` — parses saved JSON models and validates predictions within tolerance (`2e-5`)
- `Bonsai/test_pipeline.py` — trains synthetic datasets, runs full parser→generator pipeline, validates generated C behavior

## Adding a New Framework

1. Implement a `FooParser` in `Bonsai/parser.py` that returns a `UnifiedModel`
2. Add corresponding test cases in `Bonsai/test_parser.py` and `Bonsai/test_pipeline.py`
3. Add a `gen_minimal/foo_*/` directory with `Makefile` and `main.c` harness

## Precision Tuning

`EmbeddedConfig` exposes independent precision knobs for thresholds and leaf values. Tighter precision reduces header size; loosening it improves accuracy. The default is 4 decimal places for thresholds.
