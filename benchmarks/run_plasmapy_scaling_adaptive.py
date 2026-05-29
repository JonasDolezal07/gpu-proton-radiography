#!/usr/bin/env python3
"""
Adaptive-timestep scaling: prad with adaptive dt vs fixed dt=0.2 ps vs PlasmaPy.

prad's adaptive timestep scheduler uses a three-phase schedule — large dt outside
the field, small dt inside — so it completes each particle in far fewer steps than
the fixed-dt comparison.  This script measures wall time at experimentally relevant
particle counts and produces the "in practice" speedup headline.

The PlasmaPy data already comes from plasmapy_scaling.json (run run_plasmapy_scaling.py
first); this script only adds the adaptive prad timings.

Usage
-----
    python3 benchmarks/run_plasmapy_scaling_adaptive.py [--verbose]

Results saved to:
    benchmarks/results/plasmapy_scaling_adaptive.json
"""

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import utils

OUT_DIR   = utils.RESULTS_DIR
FIELD_DIR = utils.FIELDS_DIR
RUN_DIR   = utils.RESULTS_DIR / "runs"

SOURCE_X_M   = -0.080
DETECTOR_X_M =  0.100
ENERGY_MEV   = 14.7
BZ_T         = 1.0


def _ensure_field(bfld_path, n=32):
    if bfld_path.exists():
        return
    FIELD_X_MIN, FIELD_X_MAX = -0.050, 0.050
    B = np.zeros((n, n, n, 3), dtype=np.float32)
    B[:, :, :, 2] = BZ_T
    for axis in range(3):
        sz = B.shape[axis]
        taper = np.ones(sz)
        taper[0] = taper[-1] = 0.0
        taper[1] = taper[-2] = 0.5
        B *= taper.reshape([-1 if i == axis else 1 for i in range(3)] + [1])
    utils.write_bfld(bfld_path, B,
                     (FIELD_X_MIN, FIELD_X_MAX, -0.05, 0.05, -0.04, 0.04))


def _prad_deck_adaptive(bfld_path, n_particles):
    """Deck with NO dt_ps — enables prad's adaptive timestep scheduler."""
    return f"""\
[field]
path = "{bfld_path}"
scale_B = 1.0
scale_E = 0.0

[source]
type = "parallel"
direction = [1.0, 0.0, 0.0]
beam_radius_mm = 40.0
source_distance_mm = {abs(SOURCE_X_M) * 1e3:.1f}
energy_MeV = {ENERGY_MEV}
n_particles = {n_particles}

[detector]
center_mm = [{DETECTOR_X_M * 1e3:.1f}, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 200.0
height_mm = 200.0
pixels = [256, 256]

[output]
write_raw_counts = false
write_processed_counts = false
write_png = false
write_metadata = true
save_hits = false
"""


def time_prad_adaptive(bfld_path, n_particles, verbose=False):
    deck_path = RUN_DIR / f"scaling_adaptive_{n_particles}.toml"
    out_dir   = RUN_DIR / f"scaling_adaptive_{n_particles}"
    deck_path.write_text(_prad_deck_adaptive(bfld_path, n_particles))
    t0 = time.perf_counter()
    utils.run_tracer(deck_path, out_dir, verbose=verbose)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    utils.setup_dirs()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)

    bfld_path = FIELD_DIR / "scaling_uniform_Bz.bfld"
    _ensure_field(bfld_path)

    ns = [500, 1_000, 2_000, 5_000, 10_000,
          50_000, 100_000, 500_000, 1_000_000]

    records = []
    print("prad adaptive-dt runs:")
    for n in ns:
        print(f"  N={n:>9,} ... ", end="", flush=True)
        wall = time_prad_adaptive(bfld_path, n, verbose=args.verbose)
        records.append({"n": n, "wall_s": wall})
        print(f"{wall:.3f} s")

    out = OUT_DIR / "plasmapy_scaling_adaptive.json"
    out.write_text(json.dumps({"prad_adaptive": records}, indent=2))
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
