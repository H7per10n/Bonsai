"""
Parametrized generation tests.

For every (framework, task, bonsai-config) triple:
  1. Parse the stored model with UniversalParser
  2. Compare Bonsai predictions against the original saved predictions
  3. Generate the C header into tests/generated/<fw>_<task>_<cfg>/model.h
  4. Write a task-appropriate main.c alongside it

Run with:  pytest tests/test_generate.py -v
"""
import os

import numpy as np
import pytest

from Bonsai import EmbeddedConfig, MinimalEmbeddedTreeGenerator, UniversalParser
from Bonsai.model_definitions import TaskType

from .conftest import GENERATED_DIR, TASKS

# ---------------------------------------------------------------------------
# Generator configurations under test
# ---------------------------------------------------------------------------

CONFIGS = [
    ("default", EmbeddedConfig()),
    ("q16",     EmbeddedConfig(quantize=True, quantize_bits=16)),
    ("q8",      EmbeddedConfig(quantize=True, quantize_bits=8)),
]

# Max absolute error tolerance per config.
# q8 intentionally sacrifices precision for memory — wider bound is expected.
TOLERANCE = {
    "default": 1e-3,
    "q16":     1e-2,
    "q8":      1.0,
}

# ---------------------------------------------------------------------------
# main.c templates — minimal programs that exercise the generated API
# ---------------------------------------------------------------------------

_MAIN_REGRESSION = """\
#include <stdio.h>
#include <math.h>
#include "model.h"

int main(void) {
    float x[N_FEATURES] = {0};
    float result = predict(x);
    printf("predict: %f\\n", result);
    (void)result;
    return 0;
}
"""

_MAIN_BINARY = """\
#include <stdio.h>
#include <math.h>
#include "model.h"

int main(void) {
    float x[N_FEATURES] = {0};
    float prob = predict(x);
    int   cls  = predict_class(x);
    printf("predict: %f  class: %d\\n", prob, cls);
    (void)prob; (void)cls;
    return 0;
}
"""

_MAIN_MULTICLASS = """\
#include <stdio.h>
#include <math.h>
#include "model.h"

int main(void) {
    float x[N_FEATURES] = {0};
    float probs[N_CLASSES] = {0};
    predict(x, probs);
    int cls = predict_class(x);
    printf("class: %d\\n", cls);
    (void)cls;
    return 0;
}
"""

_MAIN_C = {
    TaskType.REGRESSION:              _MAIN_REGRESSION,
    TaskType.BINARY_CLASSIFICATION:   _MAIN_BINARY,
    TaskType.MULTICLASS_CLASSIFICATION: _MAIN_MULTICLASS,
}

# ---------------------------------------------------------------------------
# Parametrize: 6 model combos × 3 configs = 18 test cases
# ---------------------------------------------------------------------------

_FW_TASK = [
    ("xgb", "regression"), ("xgb", "binary"), ("xgb", "multiclass"),
    ("lgb", "regression"), ("lgb", "binary"), ("lgb", "multiclass"),
]
_CFG_IDS = [c[0] for c in CONFIGS]


@pytest.mark.parametrize("cfg_name,cfg", CONFIGS, ids=_CFG_IDS)
@pytest.mark.parametrize("fw,task", _FW_TASK, ids=[f"{f}_{t}" for f, t in _FW_TASK])
def test_generate(fw: str, task: str, cfg_name: str, cfg: EmbeddedConfig,
                  model_paths: dict, datasets: dict) -> None:
    """Parse stored model → validate predictions → write generated/ folder."""
    task_type, _ = TASKS[task]
    _, X_te, _, _ = datasets[task]
    entry = model_paths[f"{fw}_{task}"]

    # --- 1. Parse and predict with Bonsai ---
    parsed      = UniversalParser.parse(entry["json"])
    bonsai_pred = parsed.predict(X_te)

    # --- 2. Compare against original framework predictions ---
    orig_pred = np.load(entry["preds"])
    max_err   = float(np.max(np.abs(orig_pred.ravel() - bonsai_pred.ravel())))
    tol       = TOLERANCE[cfg_name]
    assert max_err <= tol, (
        f"[{fw} {task} {cfg_name}] max prediction error {max_err:.2e} exceeds tolerance {tol}"
    )

    # --- 3. Generate C header ---
    gen = MinimalEmbeddedTreeGenerator(parsed, cfg)
    gen.analyze_and_optimize()

    out_dir = os.path.join(GENERATED_DIR, f"{fw}_{task}_{cfg_name}")
    os.makedirs(out_dir, exist_ok=True)
    gen.generate_code(os.path.join(out_dir, "model.h"))

    # --- 4. Write main.c ---
    with open(os.path.join(out_dir, "main.c"), "w") as f:
        f.write(_MAIN_C[task_type])
