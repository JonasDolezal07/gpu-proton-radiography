#!/usr/bin/env python3
"""
Performance benchmark: GPU runtime vs particle count across field types.

Usage:
    python3 benchmarks/run_perf.py           # 1e5, 1e6 particles; all 4 fields
    python3 benchmarks/run_perf.py --quick   # 1e4, 1e5; zero field + z-pinch only
    python3 benchmarks/run_perf.py --full    # 1e5, 1e6, 5e6, 1e7; all 4 fields
    python3 benchmarks/run_perf.py --verbose # show tracer log output

Results saved to benchmarks/results/perf_results.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import utils

DECKS = Path(__file__).parent / "decks"

ALL_FIELDS = [
    ("zero_field",    "zero field"),
    ("zpinch",        "z-pinch"),
    ("kink_strong",   "kink"),
    ("sausage_strong","sausage"),
]
QUICK_FIELDS = ALL_FIELDS[:2]

COUNTS = {
    "quick":   [10_000, 100_000],
    "default": [100_000, 1_000_000],
    "full":    [100_000, 1_000_000, 5_000_000, 10_000_000],
}


def run_one(deck_name, field_label, n_particles, verbose):
    deck    = DECKS / f"{deck_name}.toml"
    out_dir = utils.RESULTS_DIR / "runs" / f"perf_{deck_name}_{n_particles}"
    print(f"  {field_label:12s}  n={n_particles:>10,}", end=" … ", flush=True)
    try:
        meta = utils.run_tracer(
            deck, out_dir,
            overrides={"source.n_particles": str(n_particles)},
            verbose=verbose,
        )
        runtime  = meta["performance"]["total_runtime_s"]
        diag     = meta.get("diagnostics", {})
        n_hits   = diag.get("n_hits", 0)
        gpu      = meta.get("hardware", {}).get("gpu", "unknown")
        pp_s     = n_particles / runtime
        print(f"{runtime:6.2f} s   {pp_s/1e6:.3f} Mpart/s   hits={n_hits:,}")
        return dict(
            deck=deck_name, field_label=field_label,
            n_particles=n_particles, runtime_s=runtime,
            particles_per_s=pp_s, n_hits=n_hits, gpu=gpu,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick",   action="store_true")
    ap.add_argument("--full",    action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    utils.setup_dirs()

    mode   = "quick" if args.quick else "full" if args.full else "default"
    fields = QUICK_FIELDS if args.quick else ALL_FIELDS
    counts = COUNTS[mode]

    print(f"Performance benchmark — mode={mode}")
    print(f"Fields  : {[f[1] for f in fields]}")
    print(f"Counts  : {counts}")
    print()

    results = []
    for n in counts:
        print(f"── n_particles = {n:,} ──")
        for deck_name, label in fields:
            rec = run_one(deck_name, label, n, args.verbose)
            if rec:
                results.append(rec)
        print()

    out = utils.RESULTS_DIR / "perf_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} records → {out}")


if __name__ == "__main__":
    main()
