#!/usr/bin/env python3
"""
Analytical field-grid convergence study.

Uses a Gaussian Bz filament with an exact analytical form to establish
whether the native 64×64 transverse resolution is converged — the key
limitation left open by run_convergence_grid.py, which could only compare
coarser grids against the native grid, not against an independent truth.

Field:  Bz(y, z) = B_max * exp(-(y^2 + z^2) / (2 * r0^2))
        x-independent (uniform along beam axis)
        B_max = 1.0 T,  r0 = 10 mm

Each resolution is generated independently from the same analytical formula
(not downsampled from a coarser version).  The 256×256 reference is
therefore an independent analytical truth, not just the finest available
discretisation of a single dataset.

Grids tested (NX_BEAM fixed at 8; NY = NZ = N):
    reference  256×256  r0/dx = 25.5  dx = 0.39 mm  (analytically overresolved)
               128×128  r0/dx = 12.7  dx = 0.79 mm
                64×64   r0/dx =  6.3  dx = 1.59 mm  (← "native" equivalent)
                32×32   r0/dx =  3.1  dx = 3.23 mm
                16×16   r0/dx =  1.5  dx = 6.67 mm
                 8×8    r0/dx =  0.9  dx = 14.3 mm
                 4×4    r0/dx =  0.4  dx = 33.3 mm

Domain:  x ∈ [-50, 50] mm,  y ∈ [-50, 50] mm,  z ∈ [-50, 50] mm
         (same x-span as zpinch; symmetric transverse domain)

Results:  benchmarks/results/convergence_grid_analytic.json
Plots:    docs/images/benchmark/convergence_grid_analytic.png
          docs/images/benchmark/convergence_grid_analytic_panel.png

Usage:
    python3 benchmarks/run_convergence_grid_analytic.py
    python3 benchmarks/run_convergence_grid_analytic.py --verbose
    python3 benchmarks/run_convergence_grid_analytic.py --plot-only
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import utils

RESULTS    = utils.RESULTS_DIR
PLOTS      = utils.ROOT / "docs" / "images" / "benchmark"
FIELDS_OUT = utils.FIELDS_DIR
DECK_BASE  = utils.BENCH_DIR / "decks" / "zpinch.toml"

# Analytical field parameters
B_MAX_T = 1.0
R0_MM   = 10.0

# Domain bounds in metres  (x_min, x_max, y_min, y_max, z_min, z_max)
BOUNDS = (-0.050, 0.050, -0.050, 0.050, -0.050, 0.050)

# NX_BEAM is fixed; field is x-independent so any small value gives exact trilinear.
NX_BEAM = 8
# Vary transverse NY = NZ together
NXY_VALUES    = [4, 8, 16, 32, 64, 128, 256]
REFERENCE_NXY = 256

N_PARTICLES = 500_000
DT_PS       = 0.2
MAX_STEPS   = 25_000


def _dx_mm(n):
    """Transverse grid spacing in mm for a given NY (or NZ)."""
    y_span = (BOUNDS[3] - BOUNDS[2]) * 1000.0   # mm
    return y_span / (n - 1)


def _label(n):
    return f"{NX_BEAM}x{n}x{n}"


def _bfld_path(n):
    return FIELDS_OUT / f"gauss_analytic_{n}x{n}.bfld"


def _run_dir(n):
    return RESULTS / "runs" / f"conv_gauss_{n}x{n}"


def _counts_path(n):
    return RESULTS / f"counts_gauss_{n}x{n}.npy"


# ── field generation ──────────────────────────────────────────────────────────

def _generate_bz_field(ny, nz):
    """
    Sample Bz(y, z) = B_max * exp(-(y^2 + z^2) / (2 r0^2)) analytically.
    Returns float32 array of shape (NX_BEAM, ny, nz, 3).
    """
    y_min, y_max = BOUNDS[2], BOUNDS[3]
    z_min, z_max = BOUNDS[4], BOUNDS[5]
    r0_m = R0_MM * 1e-3

    ys = np.linspace(y_min, y_max, ny)
    zs = np.linspace(z_min, z_max, nz)
    YY, ZZ = np.meshgrid(ys, zs, indexing="ij")   # (ny, nz)

    bz_2d = B_MAX_T * np.exp(-(YY**2 + ZZ**2) / (2.0 * r0_m**2))

    B = np.zeros((NX_BEAM, ny, nz, 3), dtype=np.float32)
    B[:, :, :, 2] = bz_2d[np.newaxis, :, :]   # broadcast along x
    return B


def prepare_fields():
    utils.setup_dirs()
    FIELDS_OUT.mkdir(parents=True, exist_ok=True)

    for n in NXY_VALUES:
        path = _bfld_path(n)
        B = _generate_bz_field(n, n)
        utils.write_bfld(path, B, BOUNDS)
        dx = _dx_mm(n)
        print(f"  {n:>3}×{n}  dx = {dx:.2f} mm  r0/dx = {R0_MM / dx:.1f}")


# ── simulation ────────────────────────────────────────────────────────────────

def _write_deck(n):
    """Patch the zpinch deck to point at the analytic Gaussian Bz field."""
    bfld = _bfld_path(n)
    base = DECK_BASE.read_text()
    patched = re.sub(
        r'^path\s*=.*$',
        f'path = "{bfld}"',
        base,
        flags=re.MULTILINE,
    )
    deck_out = RESULTS / f"deck_gauss_{n}x{n}.toml"
    deck_out.write_text(patched)
    return deck_out


def run_all(verbose=False):
    records = []
    for n in NXY_VALUES:
        dx  = _dx_mm(n)
        out = _run_dir(n)
        print(f"  {n:>3}×{n}  dx = {dx:.2f} mm  r0/dx = {R0_MM / dx:.1f}")

        meta = utils.run_tracer(
            _write_deck(n), out,
            overrides={
                "source.n_particles": str(N_PARTICLES),
                "numerics.dt_ps":     str(DT_PS),
                "numerics.max_steps": str(MAX_STEPS),
            },
            verbose=verbose,
        )

        counts = utils.read_raw_counts(out).astype(np.float32)
        np.save(_counts_path(n), counts)

        wall = meta["performance"]["total_runtime_s"]
        hits = meta["diagnostics"]["n_hits"]
        print(f"    hits: {hits:,}  wall: {wall:.2f}s")

        records.append(dict(
            nxy=n, dx_mm=dx, r0_over_dx=R0_MM / dx,
            n_hits=hits,
            hit_fraction=meta["diagnostics"]["hit_fraction"],
            peak_counts=float(counts.max()),
            runtime_s=wall,
        ))
    return records


def records_from_disk():
    records = []
    for n in NXY_VALUES:
        npy = _counts_path(n)
        if not npy.exists():
            print(f"Missing {npy} — run without --plot-only first")
            sys.exit(1)
        counts = np.load(npy).astype(np.float32)
        dx     = _dx_mm(n)
        meta_path = _run_dir(n) / "metadata.json"
        wall, hits, hit_frac = None, int(counts.sum()), float(counts.sum()) / N_PARTICLES
        if meta_path.exists():
            with open(meta_path) as f:
                m = json.load(f)
            wall     = m["performance"]["total_runtime_s"]
            hits     = m["diagnostics"]["n_hits"]
            hit_frac = m["diagnostics"]["hit_fraction"]
        records.append(dict(
            nxy=n, dx_mm=dx, r0_over_dx=R0_MM / dx,
            n_hits=hits, hit_fraction=hit_frac,
            peak_counts=float(counts.max()), runtime_s=wall,
        ))
    return records


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(records):
    ref      = np.load(_counts_path(REFERENCE_NXY)).astype(np.float64)
    ref_norm = ref / ref.sum()
    ref_peak = ref_norm.max()

    for r in records:
        img  = np.load(_counts_path(r["nxy"])).astype(np.float64)
        norm = img / img.sum() if img.sum() > 0 else img

        mse      = float(np.mean((ref_norm - norm) ** 2))
        corr     = float(np.corrcoef(ref_norm.ravel(), norm.ravel())[0, 1])
        peak_rel = float(norm.max() / ref_peak) if ref_peak > 0 else 1.0

        r["mse_normalised"]    = mse
        r["image_correlation"] = corr
        r["peak_fluence_rel"]  = peak_rel

    return records


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_convergence(records):
    PLOTS.mkdir(parents=True, exist_ok=True)

    dxs  = [r["dx_mm"]             for r in records]
    r0dx = [r["r0_over_dx"]        for r in records]
    mse  = [r["mse_normalised"]    for r in records]
    corr = [r["image_correlation"] for r in records]

    ref_dx = _dx_mm(REFERENCE_NXY)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(dxs, mse, "o-", color="#e05c5c", lw=2, ms=8)
    ax.axvline(ref_dx, color="#5b9bd5", ls=":", lw=1,
               label=f"reference  dx = {ref_dx:.2f} mm  (r₀/dx = {R0_MM/ref_dx:.1f})")
    native_dx = _dx_mm(64)
    ax.axvline(native_dx, color="#f0a030", ls="--", lw=1,
               label=f"native equivalent  dx = {native_dx:.2f} mm  (r₀/dx = {R0_MM/native_dx:.1f})")
    ax.set_xlabel("Grid spacing dx (mm)", fontsize=12)
    ax.set_ylabel("Normalised MSE vs 256×256 reference", fontsize=11)
    ax.set_title("Analytic-field grid convergence — MSE", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogx(dxs, corr, "o-", color="#5b9bd5", lw=2, ms=8)
    ax.axvline(ref_dx, color="#5b9bd5", ls=":", lw=1,
               label=f"reference  dx = {ref_dx:.2f} mm  (r₀/dx = {R0_MM/ref_dx:.1f})")
    ax.axvline(native_dx, color="#f0a030", ls="--", lw=1,
               label=f"native equivalent  dx = {native_dx:.2f} mm  (r₀/dx = {R0_MM/native_dx:.1f})")
    ax.axhline(1.0, color="#ccc", ls=":", lw=1)
    ax.set_xlabel("Grid spacing dx (mm)", fontsize=12)
    ax.set_ylabel("Image correlation with 256×256 reference", fontsize=11)
    ax.set_title("Analytic-field grid convergence — correlation", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Analytic Gaussian Bz filament — B_max = {B_MAX_T} T,  "
        f"r₀ = {R0_MM} mm,  N = {N_PARTICLES:,}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / "convergence_grid_analytic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_panel(records):
    n = len(records)
    ref  = np.load(_counts_path(REFERENCE_NXY)).astype(np.float32)
    vmax = float(np.log1p(ref).max())

    fig, axes = plt.subplots(1, n, figsize=(min(3.2 * n, 44), 4.0))

    for ax, r in zip(axes, records):
        nxy    = r["nxy"]
        counts = np.load(_counts_path(nxy)).astype(np.float32)
        ax.imshow(np.log1p(counts), cmap="Blues", origin="lower",
                  extent=[-250, 250, -250, 250], aspect="equal",
                  vmin=0, vmax=vmax)
        if nxy == REFERENCE_NXY:
            subtitle = "(reference)"
        else:
            subtitle = f"corr = {r['image_correlation']:.3f}"
        ax.set_title(
            f"{nxy}×{nxy}\nr₀/dx = {r['r0_over_dx']:.1f}  {subtitle}",
            fontsize=8,
        )
        ax.set_xlabel("y (mm)", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("z (mm)", fontsize=7)
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        f"Analytic Gaussian Bz — N = {N_PARTICLES:,},  dt = {DT_PS} ps",
        fontsize=11,
    )
    fig.tight_layout()
    out = PLOTS / "convergence_grid_analytic_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose",   action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip simulation; regenerate plots from saved .npy files")
    args = ap.parse_args()

    if not args.plot_only:
        print("Generating analytical Gaussian Bz field files...")
        prepare_fields()
        print(f"\nRunning {len(NXY_VALUES)} simulations  "
              f"(N = {N_PARTICLES:,},  dt = {DT_PS} ps)...")
        records = run_all(verbose=args.verbose)
    else:
        print("Loading saved results from disk...")
        records = records_from_disk()

    print("Computing metrics vs 256×256 analytical reference...")
    records = compute_metrics(records)

    print(f"\n{'nxy':>8}  {'dx (mm)':>8}  {'r0/dx':>6}  {'n_hits':>8}  "
          f"{'MSE_norm':>12}  {'corr':>10}  {'peak_rel':>9}  {'wall (s)':>9}")
    for r in records:
        wall = f"{r['runtime_s']:.2f}" if r["runtime_s"] is not None else "—"
        print(f"{r['nxy']:>8}  {r['dx_mm']:>8.2f}  {r['r0_over_dx']:>6.1f}  "
              f"{r['n_hits']:>8,}  {r['mse_normalised']:>12.3e}  "
              f"{r['image_correlation']:>10.6f}  {r['peak_fluence_rel']:>9.4f}  "
              f"{wall:>9}")

    out = RESULTS / "convergence_grid_analytic.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved → {out}")

    print("Generating plots...")
    plot_convergence(records)
    plot_panel(records)
    print("Done.")


if __name__ == "__main__":
    main()
