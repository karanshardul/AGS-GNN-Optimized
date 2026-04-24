#!/usr/bin/env python3
"""
Compare two benchmark JSON files (e.g. baseline_results.json vs optimized_results.json).

Usage:
  python compare_benchmark_json.py baseline_results.json optimized_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two AGS/LINKX benchmark JSON outputs")
    ap.add_argument("baseline", help="Path to baseline_results.json (or first run)")
    ap.add_argument("optimized", help="Path to optimized_results.json (or second run)")
    args = ap.parse_args()

    for p in (args.baseline, args.optimized):
        if not os.path.isfile(p):
            print(f"Missing file: {p}", file=sys.stderr)
            return 1

    a = load(args.baseline)
    b = load(args.optimized)

    def g(d, key, default=None):
        return d.get(key, default)

    acc_a, acc_b = g(a, "accuracy"), g(b, "accuracy")
    f1_a, f1_b = g(a, "f1"), g(b, "f1")
    pre_a, pre_b = g(a, "precompute_time"), g(b, "precompute_time")
    ep_a, ep_b = g(a, "epoch_time_avg"), g(b, "epoch_time_avg")
    tot_a, tot_b = g(a, "total_time"), g(b, "total_time")

    print("=== Benchmark comparison ===")
    print(f"Baseline:   {args.baseline}")
    print(f"Optimized:  {args.optimized}")
    print()

    if acc_a is not None and acc_b is not None:
        print(f"Accuracy:        baseline={acc_a:.6f}  optimized={acc_b:.6f}  Δ={acc_b - acc_a:+.6f}")
    if f1_a is not None and f1_b is not None:
        print(f"F1 (macro):      baseline={f1_a:.6f}  optimized={f1_b:.6f}  Δ={f1_b - f1_a:+.6f}")
    elif f1_a is None and f1_b is None:
        print("F1 (macro):      n/a (ROC-AUC or unsupported task in both runs)")
    elif f1_a is None or f1_b is None:
        print("F1 (macro):      n/a in one of the files — compare manually")

    if pre_a is not None and pre_b is not None:
        print(
            f"Precompute (s):  baseline={pre_a:.4f}  optimized={pre_b:.4f}  Δ={pre_b - pre_a:+.4f}"
        )

    if ep_a is not None and ep_b is not None and ep_b > 0:
        speedup_ep = ep_a / ep_b
        print(
            f"Epoch time avg:  baseline={ep_a:.4f}s  optimized={ep_b:.4f}s  "
            f"Δ={ep_b - ep_a:+.4f}s  speedup={speedup_ep:.3f}x"
        )

    if tot_a is not None and tot_b is not None and tot_b > 0:
        speedup_tot = tot_a / tot_b
        print(
            f"Total time (s):  baseline={tot_a:.4f}  optimized={tot_b:.4f}  "
            f"Δ={tot_b - tot_a:+.4f}s  speedup={speedup_tot:.3f}x"
        )

    # Optional extended keys
    for label, ka, kb in (
        ("Sampling time / epoch avg (s)", "sampling_time_per_epoch_avg", "sampling_time_per_epoch_avg"),
        ("Training time / epoch avg (s)", "training_time_per_epoch_avg", "training_time_per_epoch_avg"),
    ):
        va, vb = g(a, ka), g(b, kb)
        if va is not None and vb is not None:
            print(f"{label}: baseline={va:.4f}  optimized={vb:.4f}  Δ={vb - va:+.4f}")

    print()
    print("Interpretation: speedup > 1 means optimized is faster (lower time).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
