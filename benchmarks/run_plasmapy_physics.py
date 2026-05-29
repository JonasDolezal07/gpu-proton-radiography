#!/usr/bin/env python3
"""
Physics agreement: prad GPU tracer vs PlasmaPy CPU tracer.

Runs two matched test cases on both tracers and compares mean deflection,
RMS spread, and 2D hit histograms on the same detector grid.

Test cases
----------
1. uniform_Bz   — Bz = 1 T, analytic answer known (circular arc)
2. gaussian_blob — Gaussian Bz peak at origin, no simple analytic answer

Source: prad uses a 40 mm parallel beam; PlasmaPy uses the same geometry
with detector_hdir/vdir set to match prad's y/z convention.

Usage
-----
    python3 benchmarks/run_plasmapy_physics.py [--particles N] [--verbose]

Results saved to:
    benchmarks/results/plasmapy_physics.json  — statistics
    benchmarks/results/plasmapy_physics_*.npy  — 2D histograms for plotting
"""

import argparse
import json
import warnings
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import utils

OUT_DIR   = utils.RESULTS_DIR
FIELD_DIR = utils.FIELDS_DIR
RUN_DIR   = utils.RESULTS_DIR / "runs"

# Shared geometry — both tracers use these
SOURCE_X_M    = -0.080
DETECTOR_X_M  =  0.100
FIELD_X_MIN   = -0.050
FIELD_X_MAX   =  0.050
ENERGY_MEV    = 14.7
DT_PS         = 0.2
BEAM_R_MM     = 40.0

# Detector for comparison: 200 × 200 mm, 256 × 256 px
DET_HALF_MM   = 100.0
DET_BINS      = 256


# ── Field generators ──────────────────────────────────────────────────────────

def _make_uniform_Bz(path, Bz_T=1.0, n=32):
    """Uniform Bz field tapered at boundaries (PlasmaPy requires B → 0 at edge)."""
    B = np.zeros((n, n, n, 3), dtype=np.float32)
    B[:, :, :, 2] = Bz_T
    for axis in range(3):
        sz = B.shape[axis]
        taper = np.ones(sz)
        taper[0] = taper[-1] = 0.0
        taper[1] = taper[-2] = 0.5
        B *= taper.reshape([-1 if i == axis else 1 for i in range(3)] + [1])
    utils.write_bfld(path, B,
                     (FIELD_X_MIN, FIELD_X_MAX, -0.06, 0.06, -0.05, 0.05))


def _make_gaussian_blob(path, B0_T=3.0, sigma_m=0.025, n=32):
    """Gaussian Bz blob centred at origin — naturally tapers to zero at edges."""
    xs = np.linspace(FIELD_X_MIN, FIELD_X_MAX, n)
    ys = np.linspace(-0.06, 0.06, n)
    zs = np.linspace(-0.05, 0.05, n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r2 = X**2 + Y**2 + Z**2
    B = np.zeros((n, n, n, 3), dtype=np.float32)
    B[:, :, :, 2] = (B0_T * np.exp(-r2 / (2 * sigma_m**2))).astype(np.float32)
    utils.write_bfld(path, B,
                     (FIELD_X_MIN, FIELD_X_MAX, -0.06, 0.06, -0.05, 0.05))


# ── prad runner ───────────────────────────────────────────────────────────────

def _prad_deck(bfld_path, n_particles):
    return f"""\
[field]
path = "{bfld_path}"
scale_B = 1.0
scale_E = 0.0

[source]
type = "parallel"
direction = [1.0, 0.0, 0.0]
beam_radius_mm = {BEAM_R_MM}
source_distance_mm = {abs(SOURCE_X_M) * 1e3:.1f}
energy_MeV = {ENERGY_MEV}
n_particles = {n_particles}

[detector]
center_mm = [{DETECTOR_X_M * 1e3:.1f}, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = {DET_HALF_MM * 2}
height_mm = {DET_HALF_MM * 2}
pixels = [{DET_BINS}, {DET_BINS}]

[numerics]
dt_ps = {DT_PS}
max_steps = 25000

[output]
write_raw_counts = false
write_processed_counts = false
write_png = false
write_metadata = true
save_hits = true
"""


def run_prad(case_name, bfld_path, n_particles, verbose=False):
    deck_path = RUN_DIR / f"pp_physics_{case_name}.toml"
    out_dir   = RUN_DIR / f"pp_physics_{case_name}_prad"
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    deck_path.write_text(_prad_deck(bfld_path, n_particles))

    utils.run_tracer(deck_path, out_dir, verbose=verbose)

    hits = utils.read_hits_bin(out_dir)          # (N, 3): y_mm, z_mm, energy_MeV
    y_mm = hits[:, 0]
    z_mm = hits[:, 1]
    return y_mm, z_mm


# ── PlasmaPy runner ───────────────────────────────────────────────────────────

def run_plasmapy(bfld_path, n_particles, verbose=False):
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
        tracker = Tracker(
            grid, source, detector,
            dt=DT_PS * 1e-12 * u.s,
            # Match prad's detector axes: y = horizontal, z = vertical
            detector_hdir=np.array([0.0, 1.0, 0.0]),
            detector_vdir=np.array([0.0, 0.0, 1.0]),
            verbose=verbose,
        )
        tracker.create_particles(n_particles, ENERGY_MEV * u.MeV,
                                 max_theta=0.01 * u.deg, particle="p+")

    # Inject a parallel-beam disk distribution matching prad's source.
    # tracker.x starts as all-source-point (shape [N,3], metres, plain ndarray).
    # Override to uniform disk at x=SOURCE_X_M with parallel velocities.
    v_mag = float(np.linalg.norm(tracker.v[0]))
    rng = np.random.default_rng(42)
    r   = np.sqrt(rng.uniform(0.0, 1.0, n_particles)) * (BEAM_R_MM * 1e-3)
    phi = rng.uniform(0.0, 2 * np.pi, n_particles)
    tracker.x[:, 0] = SOURCE_X_M
    tracker.x[:, 1] = r * np.cos(phi)
    tracker.x[:, 2] = r * np.sin(phi)
    tracker.v[:] = 0.0
    tracker.v[:, 0] = v_mag
    tracker.theta = np.zeros(n_particles, dtype=float)  # all parallel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracker.run()

    rd = tracker.results_dict
    # With detector_hdir=[0,1,0], results["x"] is y-coord in metres
    # With detector_vdir=[0,0,1], results["y"] is z-coord in metres
    y_mm = rd["x"] * 1e3
    z_mm = rd["y"] * 1e3
    return y_mm, z_mm


# ── Statistics + histogram ────────────────────────────────────────────────────

def _stats_and_hist(y_mm, z_mm):
    """Return centroid, RMS, and 2D histogram on the comparison detector grid."""
    mean_y = float(np.mean(y_mm))
    mean_z = float(np.mean(z_mm))
    rms_y  = float(np.std(y_mm))
    rms_z  = float(np.std(z_mm))

    edges = np.linspace(-DET_HALF_MM, DET_HALF_MM, DET_BINS + 1)
    H, _, _ = np.histogram2d(y_mm, z_mm, bins=[edges, edges])
    return dict(mean_y=mean_y, mean_z=mean_z, rms_y=rms_y, rms_z=rms_z), H


def _histogram_correlation(H1, H2):
    """Pearson correlation of two normalised 2D histograms."""
    p1 = H1.ravel() / H1.sum()
    p2 = H2.ravel() / H2.sum()
    return float(np.corrcoef(p1, p2)[0, 1])


# ── Per-case runner ───────────────────────────────────────────────────────────

def run_case(case_name, bfld_path, n_particles, verbose=False):
    print(f"  [{case_name}] prad  ...", end="", flush=True)
    py, pz = run_prad(case_name, bfld_path, n_particles, verbose=verbose)
    prad_stats, prad_H = _stats_and_hist(py, pz)
    print(f" mean_y={prad_stats['mean_y']:+.2f} mm  rms_y={prad_stats['rms_y']:.2f} mm")

    print(f"  [{case_name}] PlasmaPy ...", end="", flush=True)
    qy, qz = run_plasmapy(bfld_path, n_particles, verbose=verbose)
    pp_stats, pp_H = _stats_and_hist(qy, qz)
    print(f" mean_y={pp_stats['mean_y']:+.2f} mm  rms_y={pp_stats['rms_y']:.2f} mm")

    corr = _histogram_correlation(prad_H, pp_H)
    delta_mean_y = abs(prad_stats["mean_y"] - pp_stats["mean_y"])
    delta_rms_y  = abs(prad_stats["rms_y"]  - pp_stats["rms_y"])
    print(f"  [{case_name}] Δmean_y={delta_mean_y:.2f} mm  "
          f"Δrms_y={delta_rms_y:.2f} mm  histogram_corr={corr:.4f}")

    np.save(OUT_DIR / f"pp_physics_{case_name}_prad.npy",    prad_H.astype(np.float32))
    np.save(OUT_DIR / f"pp_physics_{case_name}_plasmapy.npy", pp_H.astype(np.float32))

    return {
        "n_particles":     n_particles,
        "prad":            prad_stats,
        "plasmapy":        pp_stats,
        "delta_mean_y_mm": delta_mean_y,
        "delta_rms_y_mm":  delta_rms_y,
        "histogram_corr":  corr,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particles", type=int, default=5_000)
    ap.add_argument("--verbose",   action="store_true")
    args = ap.parse_args()

    utils.setup_dirs()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)

    n = args.particles
    print(f"Physics comparison: {n:,} particles per case")
    print()

    # Build field files
    bfld_uniform = FIELD_DIR / "pp_uniform_Bz_1T.bfld"
    bfld_blob    = FIELD_DIR / "pp_gaussian_blob.bfld"
    if not bfld_uniform.exists():
        _make_uniform_Bz(bfld_uniform)
        print("  wrote pp_uniform_Bz_1T.bfld")
    if not bfld_blob.exists():
        _make_gaussian_blob(bfld_blob)
        print("  wrote pp_gaussian_blob.bfld")

    results = {}

    print("Case 1 — Uniform Bz = 1 T (analytic answer known):")
    results["uniform_Bz"] = run_case("uniform_Bz", bfld_uniform, n,
                                     verbose=args.verbose)
    print()

    print("Case 2 — Gaussian Bz blob (peak 3 T, σ = 25 mm):")
    results["gaussian_blob"] = run_case("gaussian_blob", bfld_blob, n,
                                        verbose=args.verbose)
    print()

    out = OUT_DIR / "plasmapy_physics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Results → {out}")
    print(f"Arrays  → {OUT_DIR}/pp_physics_*.npy")


if __name__ == "__main__":
    main()
