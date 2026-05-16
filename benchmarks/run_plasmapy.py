#!/usr/bin/env python3
"""
Head-to-head comparison: PlasmaPy vs this GPU tracer.

Runs uniform Bz=1T, same geometry and energy, times both, reports speedup.
Output saved to benchmarks/plasmapy/ (gitignored).

Usage:
    python3 benchmarks/run_plasmapy.py
    python3 benchmarks/run_plasmapy.py --particles 50000
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import utils

OUT_DIR = Path(__file__).parent / "plasmapy"

SOURCE_X_M    = -0.080
DETECTOR_X_M  =  0.100
FIELD_X_MIN   = -0.050
FIELD_X_MAX   =  0.050
ENERGY_MEV    = 14.7
BZ_T          = 1.0
DT_PS         = 0.2


def steps_per_particle():
    v    = utils.proton_speed(ENERGY_MEV)
    dist = DETECTOR_X_M - SOURCE_X_M
    return int(dist / (v * DT_PS * 1e-12)) + 1


def _write_bfld_and_deck(n_particles):
    """Write field file and TOML deck for GPU run if not present."""
    bfld = OUT_DIR / "uniform_Bz_1T.bfld"
    deck = OUT_DIR / "uniform_B.toml"
    if not bfld.exists():
        B = np.zeros((8, 8, 8, 3), dtype=np.float32)
        B[:, :, :, 2] = BZ_T
        utils.write_bfld(bfld, B,
                         (FIELD_X_MIN, FIELD_X_MAX, -0.05, 0.05, -0.04, 0.04))
    if not deck.exists():
        deck.write_text(f"""[field]
path = "{bfld}"
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
width_mm = 500.0
height_mm = 500.0
pixels = [512, 512]

[numerics]
dt_ps = {DT_PS}
max_steps = 25000

[output]
write_raw_counts = true
write_processed_counts = false
write_png = false
write_metadata = true
""")
    return deck, bfld


def run_gpu(n_particles):
    deck, _ = _write_bfld_and_deck(n_particles)
    gpu_out = OUT_DIR / "gpu_run"
    t0   = time.perf_counter()
    meta = utils.run_tracer(deck, gpu_out,
                            overrides={"source.n_particles": str(n_particles)},
                            verbose=False)
    return time.perf_counter() - t0


def run_plasmapy(n_particles):
    from plasmapy.plasma.grids import CartesianGrid
    from plasmapy.diagnostics.charged_particle_radiography.synthetic_radiography import Tracker

    # 32³ grid with tapered edges (PlasmaPy requires field → 0 at boundary)
    n = 32
    xs = np.linspace(FIELD_X_MIN, FIELD_X_MAX, n)
    ys = np.linspace(-0.05, 0.05, n)
    zs = np.linspace(-0.04, 0.04, n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = CartesianGrid(X * u.m, Y * u.m, Z * u.m)

    Bz_val = np.ones((n, n, n), dtype=float)
    for axis in range(3):
        sz = Bz_val.shape[axis]
        taper = np.ones(sz)
        taper[0] = taper[-1] = 0.0
        taper[1] = taper[-2] = 0.5
        Bz_val *= taper.reshape([-1 if i == axis else 1 for i in range(3)])
    grid.add_quantities(B_z=(Bz_val * u.T))

    source   = np.array([SOURCE_X_M,   0.0, 0.0]) * u.m
    detector = np.array([DETECTOR_X_M, 0.0, 0.0]) * u.m

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracker = Tracker(grid, source, detector,
                          dt=DT_PS * 1e-12 * u.s, verbose=False)
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
    ap.add_argument("--particles", type=int, default=10_000)
    args = ap.parse_args()

    n   = args.particles
    spp = steps_per_particle()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Uniform Bz={BZ_T}T | {n:,} particles | ~{spp:,} steps/particle")
    print(f"Total particle-steps: {n * spp:,.0f}")
    print()

    print("Running GPU tracer ... ", end="", flush=True)
    gpu_wall = run_gpu(n)
    gpu_pps  = n / gpu_wall
    gpu_sps  = n * spp / gpu_wall
    print(f"{gpu_wall:.2f} s  ({gpu_pps/1e3:.1f} k particles/s  |  {gpu_sps/1e9:.2f} B steps/s)")

    print("Running PlasmaPy    ... ", end="", flush=True)
    pp_wall = run_plasmapy(n)
    pp_pps  = n / pp_wall
    pp_sps  = n * spp / pp_wall
    print(f"{pp_wall:.2f} s  ({pp_pps:.0f} particles/s  |  {pp_sps/1e6:.1f} M steps/s)")

    speedup = pp_wall / gpu_wall

    # At-scale GPU throughput from the perf benchmark (1M particle run)
    perf_json = Path(__file__).parent / "results" / "perf_results.json"
    gpu_sps_scale = gpu_sps  # fallback to measured
    if perf_json.exists():
        import json as _json
        records = _json.loads(perf_json.read_text())
        # Pick the 1M zero-field run as the GPU throughput reference
        ref = next((r for r in records
                    if r["n_particles"] == 1_000_000 and r["field_label"] == "zero field"), None)
        if ref:
            gpu_sps_scale = ref["n_particles"] * spp / ref["runtime_s"]

    speedup_scale = gpu_sps_scale / pp_sps

    print()
    print("─" * 55)
    print(f"  Measured speedup (this run, startup-dominated): {speedup:.0f}×")
    print(f"  GPU step throughput  (this run):  {gpu_sps/1e9:.2f} B steps/s")
    print(f"  GPU step throughput  (at 1M, ref):{gpu_sps_scale/1e9:.2f} B steps/s")
    print(f"  PlasmaPy step throughput:          {pp_sps/1e6:.1f} M steps/s")
    print()
    print(f"  At-scale speedup:  {speedup_scale:.0f}×")
    print(f"  A 2-hour PlasmaPy job → {2*3600/speedup_scale:.0f} s on the GPU")
    print("─" * 55)

    result = dict(
        n_particles=n, steps_per_particle=spp,
        gpu_wall_s=gpu_wall,      gpu_particles_per_s=gpu_pps,  gpu_steps_per_s=gpu_sps,
        pp_wall_s=pp_wall,        pp_particles_per_s=pp_pps,    pp_steps_per_s=pp_sps,
        speedup_wall=speedup,
    )
    out = OUT_DIR / "comparison.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
