#!/usr/bin/env python3
"""
Physics benchmark: validation cases and paraxial vs Boris comparison.

Cases:
  A  Zero field — hits should cluster at geometric projection (mean < 2 mm)
  B  Uniform Bz — measured lateral displacement follows paraxial cyclotron formula
  C  Weak z-pinch (scale_B=0.1) — Boris agrees with paraxial approximation
  D  Strong z-pinch (scale_B=1.0) — Boris diverges from paraxial (caustics form)

Usage:
    python3 benchmarks/run_physics.py           # all cases
    python3 benchmarks/run_physics.py --quick   # cases A and B only (2 B values)
    python3 benchmarks/run_physics.py --verbose

Results saved to benchmarks/results/physics_results.json.
Intermediate numpy arrays saved for plotting.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import utils

DECKS   = Path(__file__).parent / "decks"
RESULTS = utils.RESULTS_DIR
FIELDS  = utils.FIELDS_DIR
ROOT    = utils.ROOT

# Geometry shared across all physics cases — matches the benchmark decks
BEAM_RADIUS_MM    = 40.0
SOURCE_DIST_MM    = 80.0
ENERGY_MEV        = 14.7
DET_CENTER_MM     = [100.0, 0.0, 0.0]
DET_WIDTH_MM      = 500.0
DET_HEIGHT_MM     = 500.0
DET_PIXELS        = [512, 512]


def _centroid(counts):
    """Compute intensity-weighted centroid (mean_y_mm, mean_z_mm, std_y_mm, std_z_mm)."""
    W, H = counts.shape[1], counts.shape[0]
    ys = (np.arange(W) - W / 2 + 0.5) * (DET_WIDTH_MM  / W)
    zs = (np.arange(H) - H / 2 + 0.5) * (DET_HEIGHT_MM / H)
    Z, Y = np.meshgrid(zs, ys, indexing="ij")  # both (H, W)
    total = float(counts.sum())
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    w = counts / total
    my = float((w * Y).sum())
    mz = float((w * Z).sum())
    sy = float(np.sqrt((w * (Y - my)**2).sum()))
    sz = float(np.sqrt((w * (Z - mz)**2).sum()))
    return my, mz, sy, sz


# ── Case A ───────────────────────────────────────────────────────────────────

def case_A(verbose=False):
    print("Case A: zero field — straight-line projection")
    out = RESULTS / "runs" / "phys_A_zero"
    meta = utils.run_tracer(DECKS / "zero_field.toml", out,
                            overrides={"source.n_particles": "200000"},
                            verbose=verbose)
    counts = utils.read_raw_counts(out).astype(float)
    my, mz, sy, sz = _centroid(counts)
    ok = abs(my) < 2.0 and abs(mz) < 2.0
    print(f"  mean_y={my:+.2f} mm  mean_z={mz:+.2f} mm  std_y={sy:.1f} mm  std_z={sz:.1f} mm  [{'PASS' if ok else 'FAIL'}]")
    np.save(RESULTS / "counts_A.npy", counts.astype(np.float32))
    return dict(case="A", description="zero field — straight-line projection",
                n_hits=int(counts.sum()), mean_y_mm=my, mean_z_mm=mz,
                std_y_mm=sy, std_z_mm=sz, centring_ok=ok)


# ── Case B ───────────────────────────────────────────────────────────────────

def _write_uniform_bfld(path, Bz_T=1.0):
    """Uniform Bz field in x ∈ [-0.05, 0.05] m.  Outside this domain the
    shader now returns zero (explicit bounds check), so no padding needed."""
    nx = ny = nz = 8
    bounds = (-0.05, 0.05, -0.05, 0.05, -0.04, 0.04)
    B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    B[:, :, :, 2] = Bz_T
    utils.write_bfld(path, B, bounds)


def _write_uniform_deck(deck_path, bfld_path):
    txt = f"""[field]
path = "{bfld_path}"
scale_B = 1.0
scale_E = 0.0

[source]
type = "parallel"
direction = [1.0, 0.0, 0.0]
beam_radius_mm = {BEAM_RADIUS_MM}
source_distance_mm = {SOURCE_DIST_MM}
energy_MeV = {ENERGY_MEV}
n_particles = 200000

[detector]
center_mm = [{", ".join(str(x) for x in DET_CENTER_MM)}]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = {DET_WIDTH_MM}
height_mm = {DET_HEIGHT_MM}
pixels = [{", ".join(str(x) for x in DET_PIXELS)}]

[numerics]
dt_ps = 0.2
max_steps = 25000

[render]
scale = "log"
colormap = "rcf"
exposure = 1.0

[output]
write_raw_counts = true
write_processed_counts = false
write_png = false
write_metadata = true
"""
    with open(deck_path, "w") as f:
        f.write(txt)


def case_B(b_values=None, verbose=False):
    b_values = b_values or [0.1, 0.5, 1.0, 2.0]
    print(f"Case B: uniform Bz — cyclotron deflection  B = {b_values} T")

    bfld_path = FIELDS / "uniform_Bz_1T.bfld"
    if not bfld_path.exists():
        _write_uniform_bfld(bfld_path, Bz_T=1.0)
    deck_path = FIELDS / "uniform_B.toml"
    _write_uniform_deck(deck_path, bfld_path)

    p           = utils.proton_momentum(ENERGY_MEV)
    L_field     = 0.10          # field slab length [m]: x_max - x_min
    x_field_max = 0.05          # [m]
    lever       = DET_CENTER_MM[0] * 1e-3 - x_field_max   # drift from field exit to detector

    records = []
    for B0 in b_values:
        out = RESULTS / "runs" / f"phys_B_uniform_{B0:.2f}T"
        meta = utils.run_tracer(
            deck_path, out,
            overrides={"source.n_particles": "200000", "field.scale_B": str(B0)},
            verbose=verbose,
        )
        counts = utils.read_raw_counts(out).astype(float)
        my, _, _, _ = _centroid(counts)

        # Paraxial prediction (small-angle):
        #   theta = -(q/p)*Bz*L
        #   delta = theta * (L/2 + lever)   [in-field drift + post-field drift]
        theta        = -(utils.PROTON_Q / p) * B0 * L_field
        delta_parax  = theta * (L_field / 2.0 + lever) * 1e3   # mm

        # Exact circular-arc prediction for a uniform finite slab
        # (exact solution: proton follows arc of radius R = p/(qBz) inside slab,
        #  then straight drift to detector):
        R        = p / (utils.PROTON_Q * B0)
        phi      = float(np.arcsin(min(L_field / R, 1.0)))   # arc angle through slab
        dy_slab  = -R * (1.0 - np.cos(phi)) * 1e3            # displacement inside slab [mm]
        dy_drift = -np.tan(phi) * lever * 1e3                 # displacement in drift [mm]
        delta_exact = dy_slab + dy_drift

        err_parax = (my - delta_parax) / abs(delta_parax) if delta_parax != 0 else None
        err_exact = (my - delta_exact) / abs(delta_exact) if delta_exact != 0 else None

        flag = (f"err_exact={err_exact:+.1%}  err_parax={err_parax:+.1%}"
                if err_exact is not None else "")
        print(f"  B={B0:.1f} T:  measured={my:+.2f} mm  exact={delta_exact:+.2f} mm  "
              f"parax={delta_parax:+.2f} mm  {flag}")

        records.append(dict(
            B_T=B0, measured_mm=my,
            predicted_mm=delta_exact,          # primary prediction (exact circ arc)
            paraxial_mm=delta_parax,
            relative_error=err_exact,          # primary error vs exact
            relative_error_paraxial=err_parax,
            n_hits=int(counts.sum()),
        ))

    return dict(case="B", description="uniform Bz — cyclotron deflection vs exact circular arc",
                records=records)


# ── Cases C and D ─────────────────────────────────────────────────────────────

def case_CD(verbose=False):
    zpinch_bfld = ROOT / "data" / "instabilities" / "zpinch.bfld"
    results = []

    for case_id, label, scale_B in [("C", "weak (0.1×)", 0.1),
                                     ("D", "strong (1.0×)", 1.0)]:
        print(f"Case {case_id}: z-pinch {label} — Boris vs paraxial approximation")

        # Boris simulation
        out = RESULTS / "runs" / f"phys_{case_id}_zpinch"
        meta = utils.run_tracer(
            DECKS / "zpinch.toml", out,
            overrides={"source.n_particles": "500000",
                       "field.scale_B": str(scale_B)},
            verbose=verbose,
        )
        boris = utils.read_raw_counts(out).astype(np.float32)

        # Paraxial approximation — must match GPU output size (always 1024×1024)
        boris_side = int(np.sqrt(boris.size))
        paraxial, theta_y, theta_z = utils.paraxial_radiograph(
            zpinch_bfld, scale_B,
            BEAM_RADIUS_MM, DET_CENTER_MM,
            DET_WIDTH_MM, DET_HEIGHT_MM, [boris_side, boris_side], ENERGY_MEV,
            n_rays=500_000,
        )

        # Normalise paraxial to match Boris total counts
        if paraxial.sum() > 0:
            paraxial *= boris.sum() / paraxial.sum()

        corr = float(np.corrcoef(boris.ravel(), paraxial.ravel())[0, 1])
        theta_total = np.sqrt(theta_y**2 + theta_z**2)
        theta95 = float(np.degrees(np.percentile(theta_total, 95)))

        print(f"  Boris hits:  {int(boris.sum()):,}")
        print(f"  Correlation Boris vs paraxial: {corr:.4f}")
        print(f"  95th-percentile |θ|: {theta95:.2f}°")

        np.save(RESULTS / f"counts_boris_{case_id}.npy",    boris)
        np.save(RESULTS / f"counts_paraxial_{case_id}.npy", paraxial)

        results.append(dict(
            case=case_id, label=label, scale_B=scale_B,
            description=f"z-pinch {label} — Boris vs paraxial",
            boris_hits=int(boris.sum()), correlation=corr,
            theta_95th_deg=theta95,  # 95th percentile of total |theta| = sqrt(theta_y²+theta_z²)
        ))
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick",   action="store_true",
                    help="Cases A and B only (2 B values)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    utils.setup_dirs()

    all_results = {}

    res = case_A(verbose=args.verbose)
    all_results["case_A"] = res

    b_vals = [0.5, 1.0] if args.quick else [0.1, 0.5, 1.0, 2.0]
    res = case_B(b_values=b_vals, verbose=args.verbose)
    all_results["case_B"] = res

    if not args.quick:
        for r in case_CD(verbose=args.verbose):
            all_results[f"case_{r['case']}"] = r

    out = RESULTS / "physics_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
