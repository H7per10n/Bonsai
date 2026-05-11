"""
Session-scoped fixtures shared across the Bonsai test suite.

On first run, trains 6 models (XGBoost + LightGBM × 3 tasks) and saves them under
tests/models/ together with original framework predictions on the held-out test set.
Subsequent runs skip training and load directly from disk.
"""
import json
import os
import sys

import numpy as np
import pytest
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Ensure repo root is importable (redundant when root conftest.py is present, kept for safety)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bonsai.model_definitions import TaskType

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR     = os.path.dirname(__file__)
MODELS_DIR    = os.path.join(TESTS_DIR, "models")
GENERATED_DIR = os.path.join(TESTS_DIR, "generated")

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

#: Maps task name → (TaskType, n_classes)
TASKS = {
    "regression":  (TaskType.REGRESSION,                1),
    "binary":      (TaskType.BINARY_CLASSIFICATION,     2),
    "multiclass":  (TaskType.MULTICLASS_CLASSIFICATION, 4),
}

# Training hyper-parameters (kept small so training is fast)
N_ROUNDS  = 20
MAX_DEPTH = 4

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _xgb_params(task_type: TaskType, n_cls: int) -> dict:
    p = {"max_depth": MAX_DEPTH, "eta": 0.1, "seed": 42, "verbosity": 0}
    if task_type == TaskType.REGRESSION:
        p["objective"] = "reg:squarederror"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        p["objective"] = "binary:logistic"
    else:
        p["objective"] = "multi:softprob"
        p["num_class"] = n_cls
    return p


def _lgb_params(task_type: TaskType, n_cls: int) -> dict:
    p = {"max_depth": MAX_DEPTH, "learning_rate": 0.1, "verbose": -1, "seed": 42}
    if task_type == TaskType.REGRESSION:
        p["objective"] = "regression"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        p["objective"] = "binary"
    else:
        p["objective"] = "multiclass"
        p["num_class"] = n_cls
    return p

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def datasets() -> dict:
    """Download California Housing once and split into three task variants.

    Returns:
        dict mapping task name → (X_train, X_test, y_train, y_test)
    """
    print("\nFetching California Housing dataset...")
    h = fetch_california_housing()
    X     = h.data.astype(np.float32)
    y_reg = h.target.astype(np.float32)
    y_bin   = (y_reg > np.median(y_reg)).astype(np.float32)
    y_multi = np.digitize(y_reg, np.percentile(y_reg, [25, 50, 75])).astype(np.int32)

    result = {}
    for task, y in [("regression", y_reg), ("binary", y_bin), ("multiclass", y_multi)]:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        result[task] = (X_tr, X_te, y_tr, y_te)

    print(f"  {X.shape[0]:,} samples  {X.shape[1]} features")
    return result


@pytest.fixture(scope="session")
def model_paths(datasets: dict) -> dict:
    """Train (or load cached) models for all framework × task combinations.

    Saves to tests/models/:
        {fw}_{task}.json       — Bonsai-parseable JSON
        {fw}_{task}_preds.npy  — original framework predictions on X_test

    Returns:
        dict mapping "{fw}_{task}" → {"json": path, "preds": path}
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}

    for task_name, (task_type, n_cls) in TASKS.items():
        X_tr, X_te, y_tr, _ = datasets[task_name]

        for fw in ("xgb", "lgb"):
            key        = f"{fw}_{task_name}"
            json_path  = os.path.join(MODELS_DIR, f"{key}.json")
            preds_path = os.path.join(MODELS_DIR, f"{key}_preds.npy")

            if not os.path.exists(json_path) or not os.path.exists(preds_path):
                print(f"  Training {fw.upper()} {task_name}...")

                if fw == "xgb":
                    model = xgb.train(
                        _xgb_params(task_type, n_cls),
                        xgb.DMatrix(X_tr, label=y_tr),
                        num_boost_round=N_ROUNDS,
                    )
                    model.save_model(json_path)
                    orig_pred = model.predict(xgb.DMatrix(X_te))

                else:  # lgb
                    model = lgb.train(
                        _lgb_params(task_type, n_cls),
                        lgb.Dataset(X_tr, label=y_tr),
                        num_boost_round=N_ROUNDS,
                    )
                    with open(json_path, "w") as f:
                        json.dump(model.dump_model(), f)
                    orig_pred = model.predict(X_te)

                np.save(preds_path, orig_pred)

            paths[key] = {"json": json_path, "preds": preds_path}

    return paths
