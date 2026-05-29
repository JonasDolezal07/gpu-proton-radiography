#!/usr/bin/env python3
"""
Scaling comparison: prad vs PlasmaPy wall time as a function of particle count.

Both tracers run on the same uniform Bz = 1 T geometry across a range of
particle counts. The resulting two-line log-log plot shows the GPU startup
overhead at small N and the diverging throughput at large N.

PlasmaPy is run only up to --pp-max-particles (default 10,000) to keep
the total runtime reasonable (~5 min). prad is run up to 1,000,000.

Usage
-----
    python3 benchmarks/run_plasmapy_scaling.py [--pp-max-particles N] [--verbose]

Results saved to benchmarks/results/plasmapy_scaling.json.
"""

import argparse
import json
import time
import warnings
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
FIELD_X_MIN  = -0.050
FIELD_X_MAX  =  0.050
ENERGY_MEV   = 14.7
DT_PS        = 0.2
BZ_T         = 1.0


def _ensure_field(bfld_path, n=32):
    if bfld_path.exists():
        return
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


def _prad_deck(bfld_path, n_particles):
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

[numerics]
dt_ps = {DT_PS}
max_steps = 25000

[output]
write_raw_counts = false
write_processed_counts = false
write_png = false
write_metadata = true
save_hits = false
"""


def time_prad(bfld_path, n_particles, verbose=False):
    deck_path = RUN_DIR / f"scaling_prad_{n_particles}.toml"
    out_dir   = RUN_DIR / f"scaling_prad_{n_particles}"
    deck_path.write_text(_prad_deck(bfld_path, n_particles))
    t0 = time.perf_counter()
    utils.run_tracer(deck_path, out_dir, verbose=verbose)
    return time.perf_counter() - t0


def time_plasmapy(bfld_path, n_particles, verbose=False):
    import astropy.units as u
    from plasmapy.plasma.grids import CartesianGrid
    from plasmapy.diagnostics.charged_particle_radiography.synthetic_radiography import Tracker

    header, B, _ = utils.read_bfld(bfld_path)
    x_min, x_max = header["x_min"], header["x_max"]
    y_min, y_max = header["y_min"], header["y_max"]
    z_min, z_max = header["z_min"], header["z_max"]
    nx, ny, nz   = header["nx"],   header["ny"],   header["nz"]

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    zs = np.linspace(z_min, z_max, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = CartesianGrid(X * u.m, Y * u.m, Z * u.m)
    grid.add_quantities(B_z=(B[:, :, :, 2] * u.T))

    source   = np.array([SOURCE_X_M,   0.0, 0.0]) * u.m
    detector = np.array([DETECTOR_X_M, 0.0, 0.0]) * u.m

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracker = Tracker(grid, source, detector,
                          dt=DT_PS * 1e-12 * u.s, verbose=verbose)
        tracker.create_particles(n_particles, ENERGY_MEV * u.MeV,
                                 max_theta=0.01 * u.deg, particle="p+")

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracker.run()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pp-max-particles", type=int, default=10_000,
                    help="Maximum N for PlasmaPy runs (default 10,000)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    utils.setup_dirs()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)

    bfld_path = FIELD_DIR / "scaling_uniform_Bz.bfld"
    _ensure_field(bfld_path)

    prad_ns = [500, 1_000, 2_000, 5_000, 10_000,
               50_000, 100_000, 500_000, 1_000_000]
    pp_ns   = [n for n in [500, 1_000, 2_000, 5_000, 10_000]
               if n <= args.pp_max_particles]

    prad_records = []
    print("prad runs:")
    for n in prad_ns:
        print(f"  N={n:>9,} ... ", end="", flush=True)
        wall = time_prad(bfld_path, n, verbose=args.verbose)
        prad_records.append({"n": n, "wall_s": wall})
        print(f"{wall:.3f} s")

    pp_records = []
    print("\nPlasmaPy runs:")
    for n in pp_ns:
        print(f"  N={n:>9,} ... ", end="", flush=True)
        wall = time_plasmapy(bfld_path, n, verbose=args.verbose)
        pp_records.append({"n": n, "wall_s": wall})
        print(f"{wall:.1f} s")

    results = {"prad": prad_records, "plasmapy": pp_records}
    out = OUT_DIR / "plasmapy_scaling.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
