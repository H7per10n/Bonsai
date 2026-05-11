"""
Bonsai CLI  —  compile a trained XGBoost/LightGBM model to a C header.

Usage:
    python -m bonsai  model.json
    python -m bonsai  model.json  -o output.h  -m dfs  --diag

Modes:
    default   float threshold dedup, BFS layout
    q8        8-bit fixed-point thresholds, BFS
    q16       16-bit fixed-point thresholds, BFS
    dfs       float threshold dedup, DFS layout (smaller nodes)
    dfs-q8    8-bit fixed-point + DFS
    dfs-q16   16-bit fixed-point + DFS
"""
import argparse
import os
import sys
from dataclasses import replace

from . import EmbeddedConfig, MinimalEmbeddedTreeGenerator, UniversalParser

MODES = {
    "default": EmbeddedConfig(),
    "q8":      EmbeddedConfig(quantize=True, quantize_bits=8),
    "q16":     EmbeddedConfig(quantize=True, quantize_bits=16),
    "dfs":     EmbeddedConfig(dfs_layout=True),
    "dfs-q8":  EmbeddedConfig(dfs_layout=True, quantize=True, quantize_bits=8),
    "dfs-q16": EmbeddedConfig(dfs_layout=True, quantize=True, quantize_bits=16),
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m bonsai",
        description="Compile a trained XGBoost/LightGBM model to a self-contained C header.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python -m bonsai xgb_model.json -m dfs-q16 --diag",
    )
    ap.add_argument("model",
                    help="Path to trained model file (XGBoost or LightGBM JSON)")
    ap.add_argument("-o", "--out", default=None,
                    help="Output .h path  [default: <model>.h]")
    ap.add_argument("-m", "--mode", choices=MODES, default="default",
                    help="Generator mode  [default: default]")
    ap.add_argument("--no-pack", action="store_true",
                    help="Use natural struct alignment instead of #pragma pack(1)")
    ap.add_argument("--diag", action="store_true",
                    help="Print full memory diagnostic report after generation")
    args = ap.parse_args(argv)

    out = args.out or os.path.splitext(args.model)[0] + ".h"
    cfg = MODES[args.mode]
    if args.no_pack:
        cfg = replace(cfg, pack_structs=False)

    print(f"Parsing  {args.model}")
    try:
        model = UniversalParser.parse(args.model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"  {len(model.trees)} trees  "
          f"{model.num_features} features  "
          f"task={model.task_type.name}")

    gen = MinimalEmbeddedTreeGenerator(model, cfg)
    gen.analyze_and_optimize()
    gen.generate_code(out)

    m = gen.metrics
    print(f"  [{args.mode}]  "
          f"{m.original_memory.total_bytes:,} B -> {m.optimized_memory.total_bytes:,} B  "
          f"ratio={m.compression_ratio:.2f}x  node={m.node_size_reduction[1]} B")
    print(f"  Written: {out}")

    if args.diag:
        print()
        print(gen.detailed_diagnostics())

    return 0


if __name__ == "__main__":
    sys.exit(main())
