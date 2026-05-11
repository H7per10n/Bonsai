"""
Bonsai end-to-end test.

Fetches the California Housing dataset (sklearn CDN, ~1 MB), derives regression /
binary / multiclass tasks from it, trains XGBoost and LightGBM models, then runs
every Bonsai generator mode and validates that parsed predictions match the original
framework predictions within the expected tolerance.

Generated C headers land in gen_minimal/<framework>_<task>/model_<mode>.h.

Usage:
    py test_e2e.py
"""

import json
import os
import sys
import tempfile

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from bonsai import UniversalParser, MinimalEmbeddedTreeGenerator, EmbeddedConfig
from bonsai.model_definitions import TaskType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "gen_minimal")

MODES = [
    ("default",  EmbeddedConfig()),
    ("q16",      EmbeddedConfig(quantize=True, quantize_bits=16)),
    ("q8",       EmbeddedConfig(quantize=True, quantize_bits=8)),
    ("dfs",      EmbeddedConfig(dfs_layout=True)),
    ("dfs_q16",  EmbeddedConfig(dfs_layout=True, quantize=True, quantize_bits=16)),
]

TOLERANCE = {
    "default": 1e-3,
    "q16":     1e-2,
    "q8":      1.0,
    "dfs":     1e-3,
    "dfs_q16": 1e-2,
}

N_ROUNDS = 20
MAX_DEPTH = 4

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_tasks():
    print("Fetching California Housing dataset from sklearn CDN...")
    housing = fetch_california_housing()
    X, y_reg = housing.data.astype(np.float32), housing.target.astype(np.float32)
    y_bin   = (y_reg > np.median(y_reg)).astype(np.float32)
    y_multi = np.digitize(y_reg, np.percentile(y_reg, [25, 50, 75]))
    print(f"  {X.shape[0]:,} samples  {X.shape[1]} features")
    return X, y_reg, y_bin, y_multi

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgboost(X_tr, X_te, y_tr, task_type, n_classes):
    params = {"max_depth": MAX_DEPTH, "eta": 0.1, "seed": 42, "verbosity": 0}
    if task_type == TaskType.REGRESSION:
        params["objective"] = "reg:squarederror"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
    model = xgb.train(params, xgb.DMatrix(X_tr, label=y_tr), num_boost_round=N_ROUNDS)
    pred  = model.predict(xgb.DMatrix(X_te))
    return model, pred

def train_lightgbm(X_tr, X_te, y_tr, task_type, n_classes):
    params = {"max_depth": MAX_DEPTH, "learning_rate": 0.1, "verbose": -1, "seed": 42}
    if task_type == TaskType.REGRESSION:
        params["objective"] = "regression"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        params["objective"] = "binary"
    else:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes
    model = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=N_ROUNDS)
    pred  = model.predict(X_te)
    return model, pred

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_xgboost(model, path):
    model.save_model(path)

def save_lightgbm(model, path):
    with open(path, "w") as f:
        json.dump(model.dump_model(), f)

# ---------------------------------------------------------------------------
# Bonsai pipeline
# ---------------------------------------------------------------------------

def run_bonsai(model_path, X_te, orig_pred, mode_name, config, out_dir):
    """Parse → generate → validate. Returns (max_err, generator, header_path)."""
    parsed = UniversalParser.parse(model_path)
    parsed_pred = parsed.predict(X_te)

    gen = MinimalEmbeddedTreeGenerator(parsed, config)
    gen.analyze_and_optimize()

    os.makedirs(out_dir, exist_ok=True)
    header_path = os.path.join(out_dir, f"model_{mode_name}.h")
    gen.generate_code(header_path)

    max_err = float(np.max(np.abs(orig_pred.ravel() - parsed_pred.ravel())))
    return max_err, gen, header_path

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_comparison_table(results):
    """Print a side-by-side comparison of all modes across all model/task combos."""
    mode_names = [m for m, _ in MODES]

    print()
    print("=" * 100)
    print("  BONSAI MEMORY OPTIMIZATION  --  FULL COMPARISON TABLE")
    print("=" * 100)

    # Header
    col_w = 12
    print(f"  {'Model':<22}", end="")
    for m in mode_names:
        print(f"  {m:>{col_w}}", end="")
    print()

    # Subheader: bytes
    print(f"  {'(bytes optimized)':<22}", end="")
    for _ in mode_names:
        print(f"  {'bytes':>{col_w}}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in mode_names:
        print(f"  {'-'*col_w}", end="")
    print()

    # Data rows: memory bytes per mode
    for key in sorted(results.keys()):
        label = key.replace("_", " ")
        print(f"  {label:<22}", end="")
        row = results[key]
        baseline = row.get("default", {}).get("mem", None)
        for m in mode_names:
            info = row.get(m, {})
            mem = info.get("mem", None)
            if mem is None:
                print(f"  {'ERR':>{col_w}}", end="")
            else:
                print(f"  {mem:>{col_w},}", end="")
        print()

    # Ratio rows
    print()
    print(f"  {'(compression vs 24B/node)':<22}", end="")
    for _ in mode_names:
        print(f"  {'ratio x':>{col_w}}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in mode_names:
        print(f"  {'-'*col_w}", end="")
    print()

    for key in sorted(results.keys()):
        label = key.replace("_", " ")
        print(f"  {label:<22}", end="")
        row = results[key]
        for m in mode_names:
            info = row.get(m, {})
            ratio = info.get("ratio", None)
            if ratio is None:
                print(f"  {'ERR':>{col_w}}", end="")
            else:
                print(f"  {ratio:>{col_w}.2f}x", end="")
        print()

    # Node size rows
    print()
    print(f"  {'(node size B)':<22}", end="")
    for _ in mode_names:
        print(f"  {'node B':>{col_w}}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in mode_names:
        print(f"  {'-'*col_w}", end="")
    print()

    for key in sorted(results.keys()):
        label = key.replace("_", " ")
        print(f"  {label:<22}", end="")
        row = results[key]
        for m in mode_names:
            info = row.get(m, {})
            ns = info.get("node_size", None)
            if ns is None:
                print(f"  {'ERR':>{col_w}}", end="")
            else:
                print(f"  {ns:>{col_w}}", end="")
        print()

    # Max error rows
    print()
    print(f"  {'(max abs error)':<22}", end="")
    for _ in mode_names:
        print(f"  {'max_err':>{col_w}}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in mode_names:
        print(f"  {'-'*col_w}", end="")
    print()

    for key in sorted(results.keys()):
        label = key.replace("_", " ")
        print(f"  {label:<22}", end="")
        row = results[key]
        for m in mode_names:
            info = row.get(m, {})
            err = info.get("err", None)
            ok  = info.get("ok", False)
            if err is None:
                print(f"  {'ERR':>{col_w}}", end="")
            else:
                marker = "" if ok else "!"
                print(f"  {err:>{col_w-1}.2e}{marker}", end="")
        print()

    print("=" * 100)


def print_savings_summary(results):
    """Show average savings of DFS vs default across all models."""
    default_mems = []
    dfs_mems     = []
    for row in results.values():
        d = row.get("default", {}).get("mem")
        dfs = row.get("dfs", {}).get("mem")
        if d and dfs:
            default_mems.append(d)
            dfs_mems.append(dfs)

    if not default_mems:
        return

    avg_default = sum(default_mems) / len(default_mems)
    avg_dfs     = sum(dfs_mems) / len(dfs_mems)
    avg_saving  = 100.0 * (avg_default - avg_dfs) / avg_default

    print()
    print("  DFS layout vs default (averaged across all models):")
    print(f"    avg default  = {avg_default:,.0f} B")
    print(f"    avg dfs      = {avg_dfs:,.0f} B")
    print(f"    avg saving   = {avg_saving:.1f}%")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X, y_reg, y_bin, y_multi = load_tasks()

    X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_reg,   test_size=0.2, random_state=42)
    _,    _,    yb_tr, yb_te = train_test_split(X, y_bin,   test_size=0.2, random_state=42)
    _,    _,    ym_tr, ym_te = train_test_split(X, y_multi, test_size=0.2, random_state=42)

    tasks = [
        ("regression",  TaskType.REGRESSION,                yr_tr, yr_te, 1),
        ("binary",      TaskType.BINARY_CLASSIFICATION,     yb_tr, yb_te, 2),
        ("multiclass",  TaskType.MULTICLASS_CLASSIFICATION, ym_tr, ym_te, 4),
    ]

    frameworks = [
        ("xgb", "XGBoost",  train_xgboost, save_xgboost),
        ("lgb", "LightGBM", train_lightgbm, save_lightgbm),
    ]

    total_pass = total_fail = 0
    results: dict = {}        # results[model_key][mode_name] = {mem, ratio, node_size, err, ok}
    diagnostic_gen = None     # save one generator for the deep diagnostic report

    with tempfile.TemporaryDirectory() as tmpdir:
        for task_name, task_type, y_tr, y_te, n_cls in tasks:
            print(f"\n{'='*70}")
            print(f"  Task: {task_name.upper()}")
            print(f"{'='*70}")

            for fw_tag, fw_name, train_fn, save_fn in frameworks:
                print(f"\n  {fw_name}")
                model, orig_pred = train_fn(X_tr, X_te, y_tr, task_type, n_cls)

                model_path = os.path.join(tmpdir, f"{fw_tag}_{task_name}.json")
                save_fn(model, model_path)

                out_dir = os.path.join(OUTPUT_DIR, f"{fw_tag}_{task_name}")
                model_key = f"{fw_tag}_{task_name}"
                results[model_key] = {}

                for mode_name, cfg in MODES:
                    tol = TOLERANCE[mode_name]
                    try:
                        max_err, gen, hpath = run_bonsai(
                            model_path, X_te, orig_pred, mode_name, cfg, out_dir
                        )
                        ok = max_err <= tol
                        total_pass += ok
                        total_fail += not ok
                        status = "PASS" if ok else "FAIL"
                        mem   = gen.metrics.optimized_memory.total_bytes
                        ratio = gen.metrics.compression_ratio
                        ns    = gen.metrics.node_size_reduction[1]
                        results[model_key][mode_name] = {
                            "mem": mem, "ratio": ratio, "node_size": ns,
                            "err": max_err, "ok": ok,
                        }
                        print(
                            f"    [{status}] {mode_name:<8}  "
                            f"max_err={max_err:.2e}  "
                            f"memory={mem:,} B  node={ns}B  ratio={ratio:.2f}x  "
                            f"-> {os.path.relpath(hpath)}"
                        )
                        # Save XGB regression default+DFS generators for deep dive
                        if fw_tag == "xgb" and task_name == "regression" and mode_name == "dfs":
                            diagnostic_gen = gen
                    except Exception as exc:
                        total_fail += 1
                        results[model_key][mode_name] = {}
                        print(f"    [ERROR] {mode_name:<8}  {exc}")

    # ------------------------------------------------------------------
    # Performance comparison table
    # ------------------------------------------------------------------
    print_comparison_table(results)
    print_savings_summary(results)

    # ------------------------------------------------------------------
    # Deep diagnostic report for one representative model
    # ------------------------------------------------------------------
    if diagnostic_gen is not None:
        print()
        print("=" * 70)
        print("  DEEP DIAGNOSTIC  --  XGBoost Regression  (DFS layout)")
        print("=" * 70)
        print(diagnostic_gen.detailed_diagnostics())

    # ------------------------------------------------------------------
    # Overall result
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    overall = "ALL PASS" if total_fail == 0 else f"{total_fail} FAILED"
    print(f"  SUMMARY: {total_pass} passed  {total_fail} failed  ->  {overall}")
    print(f"  C headers in: {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
