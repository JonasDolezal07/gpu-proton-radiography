#!/usr/bin/env python3
"""
Field-grid resolution convergence study.

Downsamples the native z-pinch field (64×64×128) to four progressively
coarser grids and runs the standard simulation on each.  All results are
compared against the native-resolution reference to quantify how much the
radiograph degrades as field resolution decreases.

The GPU shader performs trilinear interpolation from the stored grid at each
particle step, so this study directly tests whether the native grid is fine
enough for the interpolation to accurately represent the underlying field.

Grids tested:
    native  64×64×128  dx = 1.587 mm  (reference)
    ×0.5    32×32×64   dx = 3.226 mm
    ×0.25   16×16×32   dx = 6.667 mm
    ×0.125   8×8×16    dx = 14.29 mm
    ×0.0625  4×4×8     dx = 33.33 mm

Results:  benchmarks/results/convergence_grid.json
Plots:    docs/images/benchmark/convergence_grid.png
          docs/images/benchmark/convergence_grid_panel.png

Usage:
    python3 benchmarks/run_convergence_grid.py
    python3 benchmarks/run_convergence_grid.py --verbose
    python3 benchmarks/run_convergence_grid.py --plot-only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).parent))
import utils

RESULTS     = utils.RESULTS_DIR
PLOTS       = utils.ROOT / "docs" / "images" / "benchmark"
NATIVE_BFLD = utils.ROOT / "data" / "instabilities" / "zpinch.bfld"
DECK        = utils.BENCH_DIR / "decks" / "zpinch.toml"
FIELDS_OUT  = utils.FIELDS_DIR   # benchmarks/fields/

# (nx, ny, nz) — native is the reference
RESOLUTIONS = [
    (4,   4,   8),
    (8,   8,  16),
    (16,  16,  32),
    (32,  32,  64),
    (64,  64, 128),   # native — reference
]
REFERENCE_RES = (64, 64, 128)

N_PARTICLES = 500_000
DT_PS       = 0.2
MAX_STEPS   = 25_000


def _label(res):
    return f"{res[0]}x{res[1]}x{res[2]}"


def _bfld_path(res):
    return FIELDS_OUT / f"zpinch_grid_{_label(res)}.bfld"


def _run_dir(res):
    return RESULTS / "runs" / f"conv_grid_{_label(res)}"


def _counts_path(res):
    return RESULTS / f"counts_grid_{_label(res)}.npy"


def _dx_mm(nx, header):
    """Grid spacing in x for a given nx, in mm."""
    span = header["x_max"] - header["x_min"]
    return span / (nx - 1) * 1000.0


# ── field preparation ─────────────────────────────────────────────────────────

def prepare_fields():
    """Downsample the native field to each tested resolution and write .bfld files."""
    utils.setup_dirs()
    FIELDS_OUT.mkdir(parents=True, exist_ok=True)

    header, B_native, _ = utils.read_bfld(NATIVE_BFLD)
    bounds = (header["x_min"], header["x_max"],
              header["y_min"], header["y_max"],
              header["z_min"], header["z_max"])
    nx_nat, ny_nat, nz_nat = header["nx"], header["ny"], header["nz"]

    for res in RESOLUTIONS:
        nx, ny, nz = res
        path = _bfld_path(res)

        if nx == nx_nat and ny == ny_nat and nz == nz_nat:
            # Native resolution — copy as-is (or just link to original path)
            utils.write_bfld(path, B_native, bounds)
            print(f"  {_label(res)}  (native — copied as-is)")
            continue

        # Downsample using bilinear interpolation (order=1), consistent with
        # what the GPU trilinear sampler does when reading the stored field.
        factors = (nx / nx_nat, ny / ny_nat, nz / nz_nat, 1.0)
        B_down  = zoom(B_native, factors, order=1)
        utils.write_bfld(path, B_down.astype(np.float32), bounds)

        dx = _dx_mm(nx, header)
        print(f"  {_label(res)}  dx = {dx:.2f} mm")


# ── simulation ────────────────────────────────────────────────────────────────

def _write_deck(res):
    """Write a copy of the zpinch deck pointing at the downsampled field."""
    bfld = _bfld_path(res)
    base = DECK.read_text()
    # Replace the field path line — works for any absolute or relative path.
    import re
    patched = re.sub(
        r'^path\s*=.*$',
        f'path = "{bfld}"',
        base,
        flags=re.MULTILINE,
    )
    deck_out = RESULTS / f"deck_grid_{_label(res)}.toml"
    deck_out.write_text(patched)
    return deck_out


def run_all(verbose=False):
    header = utils.read_bfld(NATIVE_BFLD)[0]
    records = []
    for res in RESOLUTIONS:
        out  = _run_dir(res)
        dx   = _dx_mm(res[0], header)
        deck = _write_deck(res)
        print(f"  {_label(res)}  dx = {dx:.2f} mm")

        meta = utils.run_tracer(
            deck, out,
            overrides={
                "source.n_particles":  str(N_PARTICLES),
                "numerics.dt_ps":      str(DT_PS),
                "numerics.max_steps":  str(MAX_STEPS),
            },
            verbose=verbose,
        )

        counts = utils.read_raw_counts(out).astype(np.float32)
        np.save(_counts_path(res), counts)

        wall = meta["performance"]["total_runtime_s"]
        hits = meta["diagnostics"]["n_hits"]
        print(f"    hits: {hits:,}  wall: {wall:.2f}s")

        records.append(dict(
            resolution=_label(res),
            nx=res[0], ny=res[1], nz=res[2],
            dx_mm=dx,
            n_hits=hits,
            hit_fraction=meta["diagnostics"]["hit_fraction"],
            peak_counts=float(counts.max()),
            runtime_s=wall,
        ))

    return records


def records_from_disk():
    header = utils.read_bfld(NATIVE_BFLD)[0]
    records = []
    for res in RESOLUTIONS:
        npy = _counts_path(res)
        if not npy.exists():
            print(f"Missing {npy} — run without --plot-only first")
            sys.exit(1)
        counts = np.load(npy).astype(np.float32)
        meta_path = _run_dir(res) / "metadata.json"
        wall, hits, hit_frac = None, int(counts.sum()), float(counts.sum()) / N_PARTICLES
        if meta_path.exists():
            with open(meta_path) as f:
                m = json.load(f)
            wall     = m["performance"]["total_runtime_s"]
            hits     = m["diagnostics"]["n_hits"]
            hit_frac = m["diagnostics"]["hit_fraction"]
        records.append(dict(
            resolution=_label(res), nx=res[0], ny=res[1], nz=res[2],
            dx_mm=_dx_mm(res[0], header),
            n_hits=hits, hit_fraction=hit_frac,
            peak_counts=float(counts.max()), runtime_s=wall,
        ))
    return records


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(records):
    ref = np.load(_counts_path(REFERENCE_RES)).astype(np.float64)
    ref_norm = ref / ref.sum()
    ref_peak = ref_norm.max()

    for r in records:
        res  = (r["nx"], r["ny"], r["nz"])
        img  = np.load(_counts_path(res)).astype(np.float64)
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

    dxs  = [r["dx_mm"]            for r in records]
    mse  = [r["mse_normalised"]   for r in records]
    corr = [r["image_correlation"] for r in records]

    ref_dx = _dx_mm(REFERENCE_RES[0], utils.read_bfld(NATIVE_BFLD)[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(dxs, mse, "o-", color="#e05c5c", lw=2, ms=8)
    ax.axvline(ref_dx, color="#5b9bd5", ls=":", lw=1,
               label=f"native grid  dx = {ref_dx:.2f} mm")
    ax.set_xlabel("Grid spacing dx (mm)", fontsize=12)
    ax.set_ylabel("Normalised MSE vs native-resolution reference", fontsize=11)
    ax.set_title("Grid-resolution convergence — MSE", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogx(dxs, corr, "o-", color="#5b9bd5", lw=2, ms=8)
    ax.axvline(ref_dx, color="#5b9bd5", ls=":", lw=1,
               label=f"native grid  dx = {ref_dx:.2f} mm")
    ax.axhline(1.0, color="#ccc", ls=":", lw=1)
    ax.set_xlabel("Grid spacing dx (mm)", fontsize=12)
    ax.set_ylabel("Image correlation with native reference", fontsize=11)
    ax.set_title("Grid-resolution convergence — correlation", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Field-grid resolution convergence — z-pinch,  "
        f"N = {N_PARTICLES:,},  dt = {DT_PS} ps",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / "convergence_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_panel(records):
    n = len(records)
    ref = np.load(_counts_path(REFERENCE_RES)).astype(np.float32)
    vmax = float(np.log1p(ref).max())

    fig, axes = plt.subplots(1, n, figsize=(min(3.2 * n, 44), 4.0))

    for ax, r in zip(axes, records):
        res    = (r["nx"], r["ny"], r["nz"])
        counts = np.load(_counts_path(res)).astype(np.float32)
        ax.imshow(np.log1p(counts), cmap="Blues", origin="lower",
                  extent=[-250, 250, -250, 250], aspect="equal",
                  vmin=0, vmax=vmax)
        if res == REFERENCE_RES:
            subtitle = "(native reference)"
        else:
            subtitle = f"corr = {r['image_correlation']:.3f}"
        ax.set_title(f"{r['resolution']}\ndx={r['dx_mm']:.1f}mm  {subtitle}", fontsize=8)
        ax.set_xlabel("y (mm)", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("z (mm)", fontsize=7)
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        f"Radiographs at each grid resolution — z-pinch,  "
        f"N = {N_PARTICLES:,},  dt = {DT_PS} ps",
        fontsize=11,
    )
    fig.tight_layout()
    out = PLOTS / "convergence_grid_panel.png"
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
        print("Preparing downsampled field files...")
        prepare_fields()
        print(f"\nRunning {len(RESOLUTIONS)} simulations  "
              f"(N = {N_PARTICLES:,},  dt = {DT_PS} ps)...")
        records = run_all(verbose=args.verbose)
    else:
        print("Loading saved results from disk...")
        records = records_from_disk()

    print("Computing metrics vs native reference...")
    records = compute_metrics(records)

    print(f"\n{'resolution':>14}  {'dx (mm)':>8}  {'n_hits':>8}  "
          f"{'MSE_norm':>12}  {'corr':>10}  {'peak_rel':>9}  {'wall (s)':>9}")
    for r in records:
        wall = f"{r['runtime_s']:.2f}" if r["runtime_s"] is not None else "—"
        print(f"{r['resolution']:>14}  {r['dx_mm']:>8.2f}  {r['n_hits']:>8,}  "
              f"{r['mse_normalised']:>12.3e}  {r['image_correlation']:>10.6f}  "
              f"{r['peak_fluence_rel']:>9.4f}  {wall:>9}")

    out = RESULTS / "convergence_grid.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved → {out}")

    print("Generating plots...")
    plot_convergence(records)
    plot_panel(records)
    print("Done.")


if __name__ == "__main__":
    main()
