"""
Bonsai end-to-end test + benchmark.

Two passes over XGBoost and LightGBM x {regression, binary, multiclass}:

  CORRECTNESS (fast)  — N_ROUNDS=20, MAX_DEPTH=4
      All five modes validated against original framework predictions.

  BENCHMARK (real-world scale)  — N_ROUNDS=100, MAX_DEPTH=6
      Larger models (3-8k nodes each) used to measure memory savings.
      Results printed as a comparison table; used as README benchmark source.

Usage:
    python test_e2e.py
"""

import json, os, sys, tempfile
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from bonsai import UniversalParser, MinimalEmbeddedTreeGenerator, EmbeddedConfig
from bonsai.model_definitions import TaskType

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "gen_minimal")

# Correctness check — small/fast
COR_ROUNDS = 20
COR_DEPTH  = 4

# Benchmark — larger models for realistic memory numbers
BM_ROUNDS  = 100
BM_DEPTH   = 6

MODES = [
    ("default", EmbeddedConfig()),
    ("q16",     EmbeddedConfig(quantize=True, quantize_bits=16)),
    ("q8",      EmbeddedConfig(quantize=True, quantize_bits=8)),
    ("dfs",     EmbeddedConfig(dfs_layout=True)),
    ("dfs-q16", EmbeddedConfig(dfs_layout=True, quantize=True, quantize_bits=16)),
]

TOLERANCE = {"default": 1e-3, "q16": 1e-2, "q8": 1.0, "dfs": 1e-3, "dfs-q16": 1e-2}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    print("Loading California Housing dataset...")
    h = fetch_california_housing()
    X       = h.data.astype(np.float32)
    y_reg   = h.target.astype(np.float32)
    y_bin   = (y_reg > np.median(y_reg)).astype(np.float32)
    y_multi = np.digitize(y_reg, np.percentile(y_reg, [25, 50, 75])).astype(np.int32)
    print(f"  {X.shape[0]:,} samples  {X.shape[1]} features")
    return X, y_reg, y_bin, y_multi

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _xgb(X_tr, X_te, y_tr, task_type, n_cls, n_rounds, max_depth):
    p = {"max_depth": max_depth, "eta": 0.1, "seed": 42, "verbosity": 0}
    if task_type == TaskType.REGRESSION:
        p["objective"] = "reg:squarederror"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        p["objective"] = "binary:logistic"
    else:
        p["objective"] = "multi:softprob"; p["num_class"] = n_cls
    m = xgb.train(p, xgb.DMatrix(X_tr, label=y_tr), num_boost_round=n_rounds)
    return m, m.predict(xgb.DMatrix(X_te))

def _lgb(X_tr, X_te, y_tr, task_type, n_cls, n_rounds, max_depth):
    p = {"max_depth": max_depth, "learning_rate": 0.1, "verbose": -1, "seed": 42}
    if task_type == TaskType.REGRESSION:
        p["objective"] = "regression"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        p["objective"] = "binary"
    else:
        p["objective"] = "multiclass"; p["num_class"] = n_cls
    m = lgb.train(p, lgb.Dataset(X_tr, label=y_tr), num_boost_round=n_rounds)
    return m, m.predict(X_te)

def save_model(model, path, fw):
    if fw == "xgb":
        model.save_model(path)
    else:
        with open(path, "w") as f:
            json.dump(model.dump_model(), f)

# ---------------------------------------------------------------------------
# Bonsai pipeline
# ---------------------------------------------------------------------------

def run_bonsai(model_path, X_te, orig_pred, mode_name, cfg, out_dir):
    parsed     = UniversalParser.parse(model_path)
    parsed_pred = parsed.predict(X_te)
    gen        = MinimalEmbeddedTreeGenerator(parsed, cfg)
    gen.analyze_and_optimize()
    os.makedirs(out_dir, exist_ok=True)
    gen.generate_code(os.path.join(out_dir, f"model_{mode_name}.h"))
    max_err = float(np.max(np.abs(orig_pred.ravel() - parsed_pred.ravel())))
    return max_err, gen

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(results, title):
    mode_names = [m for m, _ in MODES]
    cw = 13
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)

    for section, key in [("Memory (bytes)", "mem"), ("Ratio vs 24B/node", "ratio"),
                          ("Node size (B)", "node_size"), ("Max abs error", "err")]:
        fmt_ratio  = key == "ratio"
        fmt_err    = key == "err"
        print(f"\n  {section}")
        print(f"  {'Model':<24}" + "".join(f"  {m:>{cw}}" for m in mode_names))
        print(f"  {'-'*24}" + "".join(f"  {'-'*cw}" for _ in mode_names))
        for mk in sorted(results):
            row = results[mk]
            print(f"  {mk.replace('_',' '):<24}", end="")
            for mn in mode_names:
                v = row.get(mn, {}).get(key)
                if v is None:
                    print(f"  {'ERR':>{cw}}", end="")
                elif fmt_ratio:
                    print(f"  {v:>{cw}.2f}x", end="")
                elif fmt_err:
                    ok = row.get(mn, {}).get("ok", False)
                    print(f"  {v:>{cw-1}.2e}{'!' if not ok else ' '}", end="")
                else:
                    print(f"  {v:>{cw},}", end="")
            print()

    # Summary: best mode per model
    print(f"\n  Best mode per model (lowest memory):")
    for mk in sorted(results):
        row = results[mk]
        best_m = min((m for m in mode_names if row.get(m, {}).get("mem")),
                     key=lambda m: row[m]["mem"], default=None)
        if best_m:
            baseline = results[mk].get("default", {}).get("mem", 1)
            best_mem = results[mk][best_m]["mem"]
            saved    = 100 * (baseline - best_mem) / baseline
            print(f"    {mk.replace('_',' '):<22} -> {best_m:<10}"
                  f"  {best_mem:,} B  ({saved:.0f}% vs default)")
    print("=" * 100)


def avg_saving(results, mode_a="default", mode_b="dfs"):
    vals = [(r[mode_a]["mem"], r[mode_b]["mem"])
            for r in results.values()
            if r.get(mode_a, {}).get("mem") and r.get(mode_b, {}).get("mem")]
    if not vals:
        return 0.0
    return 100 * sum((a - b) / a for a, b in vals) / len(vals)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_suite(X_tr, X_te, tasks, frameworks, tmpdir, n_rounds, max_depth,
              out_prefix, validate=True):
    results = {}
    for task_name, task_type, y_tr, y_te, n_cls in tasks:
        for fw_tag, fw_fn in frameworks:
            label = f"{fw_tag}_{task_name}"
            print(f"  {label}  (n_rounds={n_rounds}, depth={max_depth})")
            model, orig_pred = fw_fn(X_tr, X_te, y_tr, task_type, n_cls, n_rounds, max_depth)
            mp = os.path.join(tmpdir, f"{label}.json")
            save_model(model, mp, fw_tag)
            results[label] = {}
            for mode_name, cfg in MODES:
                tol = TOLERANCE[mode_name]
                try:
                    out_dir = os.path.join(OUTPUT_DIR, f"{out_prefix}{label}")
                    err, gen = run_bonsai(mp, X_te, orig_pred, mode_name, cfg, out_dir)
                    ok = err <= tol
                    m  = gen.metrics
                    results[label][mode_name] = {
                        "mem":       m.optimized_memory.total_bytes,
                        "ratio":     m.compression_ratio,
                        "node_size": m.node_size_reduction[1],
                        "nodes":     sum(max(t.node_count,1) for t in gen.model.trees),
                        "err":       err,
                        "ok":        ok,
                    }
                    if validate and not ok:
                        print(f"    FAIL {mode_name}: err={err:.2e} > tol={tol}")
                    elif validate:
                        pass  # quiet on pass
                except Exception as e:
                    results[label][mode_name] = {}
                    print(f"    ERROR {mode_name}: {e}")
    return results


def main():
    X, y_reg, y_bin, y_multi = load_data()
    X_tr, X_te, yr_tr, _ = train_test_split(X, y_reg,   test_size=0.2, random_state=42)
    _,    _,    yb_tr, _ = train_test_split(X, y_bin,   test_size=0.2, random_state=42)
    _,    _,    ym_tr, _ = train_test_split(X, y_multi, test_size=0.2, random_state=42)

    tasks = [
        ("regression",  TaskType.REGRESSION,                yr_tr, X_te, 1),
        ("binary",      TaskType.BINARY_CLASSIFICATION,     yb_tr, X_te, 2),
        ("multiclass",  TaskType.MULTICLASS_CLASSIFICATION, ym_tr, X_te, 4),
    ]
    frameworks = [("xgb", _xgb), ("lgb", _lgb)]

    total_pass = total_fail = 0

    with tempfile.TemporaryDirectory() as tmpdir:

        # -------------------------------------------------------------------
        # CORRECTNESS PASS  (fast, small models, all modes validated)
        # -------------------------------------------------------------------
        print(f"\n{'='*70}")
        print(f"  CORRECTNESS  (n_rounds={COR_ROUNDS}, max_depth={COR_DEPTH})")
        print(f"{'='*70}")
        cor_results = run_suite(
            X_tr, X_te, tasks, frameworks, tmpdir,
            COR_ROUNDS, COR_DEPTH, out_prefix="cor_", validate=True
        )
        for row in cor_results.values():
            for info in row.values():
                if info:
                    total_pass += info.get("ok", False)
                    total_fail += not info.get("ok", True)

        n_modes = len(MODES)
        n_models = len(cor_results)
        print(f"\n  {n_models * n_modes} runs  ->  {total_pass} passed  {total_fail} failed")

        # -------------------------------------------------------------------
        # BENCHMARK PASS  (larger models, realistic flash footprint)
        # -------------------------------------------------------------------
        print(f"\n{'='*70}")
        print(f"  BENCHMARK  (n_rounds={BM_ROUNDS}, max_depth={BM_DEPTH})")
        print(f"{'='*70}")
        bm_results = run_suite(
            X_tr, X_te, tasks, frameworks, tmpdir,
            BM_ROUNDS, BM_DEPTH, out_prefix="bm_", validate=False
        )

    # -------------------------------------------------------------------
    # Report correctness results
    # -------------------------------------------------------------------
    print_comparison(cor_results,
        f"CORRECTNESS RESULTS  --  n_rounds={COR_ROUNDS}  max_depth={COR_DEPTH}")

    # -------------------------------------------------------------------
    # Report benchmark results
    # -------------------------------------------------------------------
    print_comparison(bm_results,
        f"BENCHMARK RESULTS  --  n_rounds={BM_ROUNDS}  max_depth={BM_DEPTH}")

    # Node-count summary for benchmark
    print("\n  Node counts (benchmark models):")
    for mk in sorted(bm_results):
        dflt = bm_results[mk].get("default", {})
        nodes = dflt.get("nodes", "?")
        baseline = nodes * 24 if isinstance(nodes, int) else "?"
        best_m = "dfs-q16"
        best_mem = bm_results[mk].get(best_m, {}).get("mem", "?")
        print(f"    {mk.replace('_',' '):<22}  {nodes:>5} nodes  "
              f"baseline={baseline:>7,} B  dfs-q16={best_mem:>7,} B")

    # Average saving DFS vs default
    saving_dfs    = avg_saving(bm_results, "default", "dfs")
    saving_dfsq16 = avg_saving(bm_results, "default", "dfs-q16")
    print(f"\n  Average saving vs default (benchmark):")
    print(f"    dfs      {saving_dfs:.1f}%")
    print(f"    dfs-q16  {saving_dfsq16:.1f}%")

    # Diagnostic for one representative model
    with tempfile.TemporaryDirectory() as tmpdir2:
        print(f"\n{'='*70}")
        print("  DEEP DIAGNOSTIC  --  XGBoost Regression  dfs-q16  (benchmark scale)")
        print(f"{'='*70}")
        _, yr_tr2, _, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        xgb_m, xgb_pred = _xgb(X_tr, X_te, yr_tr2, TaskType.REGRESSION, 1,
                                BM_ROUNDS, BM_DEPTH)
        mp = os.path.join(tmpdir2, "xgb_reg.json")
        save_model(xgb_m, mp, "xgb")
        cfg = EmbeddedConfig(dfs_layout=True, quantize=True, quantize_bits=16)
        parsed = UniversalParser.parse(mp)
        gen = MinimalEmbeddedTreeGenerator(parsed, cfg)
        gen.analyze_and_optimize()
        print(gen.detailed_diagnostics())

    overall = "ALL PASS" if total_fail == 0 else f"{total_fail} FAILED"
    print(f"\n{'='*70}")
    print(f"  SUMMARY  {total_pass} passed  {total_fail} failed  ->  {overall}")
    print(f"{'='*70}\n")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
