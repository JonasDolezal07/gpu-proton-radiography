#!/usr/bin/env python3
"""
Timestep convergence study for the Boris integrator.

Runs the standard z-pinch benchmark across nine dt values spanning three
decades (0.05–12.8 ps) with a fixed total simulation time budget. All results
are compared against the finest-dt run (reference) using normalised MSE and
pixel-level image correlation.

Expected result: a converged plateau at fine dt transitioning to degraded
accuracy at coarse dt, with the default dt = 0.2 ps comfortably within the
plateau.

Results:  benchmarks/results/convergence_dt.json
Plots:    docs/images/benchmark/convergence_dt.png
          docs/images/benchmark/convergence_dt_panel.png

Usage:
    python3 benchmarks/run_convergence_dt.py
    python3 benchmarks/run_convergence_dt.py --verbose
    python3 benchmarks/run_convergence_dt.py --plot-only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import utils

RESULTS = utils.RESULTS_DIR
PLOTS   = utils.ROOT / "docs" / "images" / "benchmark"
DECK    = utils.BENCH_DIR / "decks" / "zpinch.toml"

# dt values tested, coarsest to finest.  Finest is the reference.
DT_PS_VALUES  = [204.8, 102.4, 51.2, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2, 0.1, 0.05]
REFERENCE_DT  = 0.05      # ps — finest dt, used as ground-truth image
TOTAL_TIME_PS = 5000.0    # ps — fixed budget for all runs (matches default deck)
N_PARTICLES   = 500_000


def _max_steps(dt_ps):
    # Add 25% margin over the nominal time budget so strongly-deflected
    # particles are not step-starved at coarse dt values.
    return int(TOTAL_TIME_PS * 1.25 / dt_ps)


def _run_dir(dt_ps):
    return RESULTS / "runs" / f"conv_dt_{dt_ps:.2f}ps"


def _counts_path(dt_ps):
    return RESULTS / f"counts_dt_{dt_ps:.2f}ps.npy"


# ── simulation ────────────────────────────────────────────────────────────────

def run_all(verbose=False):
    utils.setup_dirs()
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)

    records = []
    for dt in DT_PS_VALUES:
        ms = _max_steps(dt)
        out = _run_dir(dt)
        print(f"  dt = {dt:.2f} ps  max_steps = {ms:,}")

        meta = utils.run_tracer(
            DECK, out,
            overrides={
                "source.n_particles": str(N_PARTICLES),
                "numerics.dt_ps":     str(dt),
                "numerics.max_steps": str(ms),
            },
            verbose=verbose,
        )

        counts = utils.read_raw_counts(out).astype(np.float32)
        np.save(_counts_path(dt), counts)

        wall = meta["performance"]["total_runtime_s"]
        hits = meta["diagnostics"]["n_hits"]
        print(f"    hits: {hits:,}  wall: {wall:.2f}s")

        records.append(dict(
            dt_ps=dt,
            max_steps=ms,
            total_time_ps=TOTAL_TIME_PS,
            n_particles=N_PARTICLES,
            n_hits=hits,
            hit_fraction=meta["diagnostics"]["hit_fraction"],
            peak_counts=float(counts.max()),
            runtime_s=wall,
        ))

    return records


def records_from_disk():
    """Reconstruct records from saved .npy files and run metadata."""
    records = []
    for dt in DT_PS_VALUES:
        npy = _counts_path(dt)
        if not npy.exists():
            print(f"Missing {npy} — run without --plot-only first")
            sys.exit(1)
        counts = np.load(npy).astype(np.float32)
        meta_path = _run_dir(dt) / "metadata.json"
        wall, hits, hit_frac = None, int(counts.sum()), float(counts.sum()) / N_PARTICLES
        if meta_path.exists():
            with open(meta_path) as f:
                m = json.load(f)
            wall     = m["performance"]["total_runtime_s"]
            hits     = m["diagnostics"]["n_hits"]
            hit_frac = m["diagnostics"]["hit_fraction"]
        records.append(dict(
            dt_ps=dt, max_steps=_max_steps(dt),
            total_time_ps=TOTAL_TIME_PS, n_particles=N_PARTICLES,
            n_hits=hits, hit_fraction=hit_frac,
            peak_counts=float(counts.max()), runtime_s=wall,
        ))
    return records


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(records):
    ref_path = _counts_path(REFERENCE_DT)
    ref = np.load(ref_path).astype(np.float64)
    ref_total = ref.sum()
    ref_norm  = ref / ref_total if ref_total > 0 else ref
    ref_peak  = ref.max()

    for r in records:
        img   = np.load(_counts_path(r["dt_ps"])).astype(np.float64)
        total = img.sum()
        norm  = img / total if total > 0 else img

        # Normalised MSE: compare probability-density images so absolute hit
        # count differences (from small step-budget edge effects) don't dominate.
        mse  = float(np.mean((ref_norm - norm) ** 2))
        corr = float(np.corrcoef(ref_norm.ravel(), norm.ravel())[0, 1])
        # Peak fluence relative to reference (same normalisation)
        peak_rel = float(norm.max() / ref_norm.max()) if ref_norm.max() > 0 else 1.0

        r["mse_normalised"]    = mse
        r["image_correlation"] = corr
        r["peak_fluence_rel"]  = peak_rel

    return records


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_convergence(records):
    PLOTS.mkdir(parents=True, exist_ok=True)

    dts  = [r["dt_ps"]            for r in records]
    mse  = [r["mse_normalised"]   for r in records]
    corr = [r["image_correlation"] for r in records]
    peak = [r["peak_fluence_rel"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.loglog(dts, mse, "o-", color="#e05c5c", lw=2, ms=8)
    ax.axvline(REFERENCE_DT, color="#aaa", ls="--", lw=1,
               label=f"reference  dt = {REFERENCE_DT} ps")
    ax.set_xlabel("dt (ps)", fontsize=12)
    ax.set_ylabel("Normalised MSE vs reference", fontsize=11)
    ax.set_title("MSE convergence", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    ax = axes[1]
    ax.semilogx(dts, corr, "o-", color="#5b9bd5", lw=2, ms=8)
    ax.axvline(REFERENCE_DT, color="#aaa", ls="--", lw=1,
               label=f"reference  dt = {REFERENCE_DT} ps")
    ax.axhline(1.0, color="#ccc", ls=":", lw=1)
    ax.set_xlabel("dt (ps)", fontsize=12)
    ax.set_ylabel("Image correlation with reference", fontsize=11)
    ax.set_title("Image correlation convergence", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    ax = axes[2]
    ax.semilogx(dts, peak, "o-", color="#5dbf5d", lw=2, ms=8)
    ax.axvline(REFERENCE_DT, color="#aaa", ls="--", lw=1,
               label=f"reference  dt = {REFERENCE_DT} ps")
    ax.axhline(1.0, color="#ccc", ls=":", lw=1)
    ax.set_xlabel("dt (ps)", fontsize=12)
    ax.set_ylabel("Peak fluence relative to reference", fontsize=11)
    ax.set_title("Peak fluence convergence", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    fig.suptitle(
        f"Timestep convergence — z-pinch field,  {N_PARTICLES:,} particles,  "
        f"total time = {TOTAL_TIME_PS:.0f} ps",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / "convergence_dt.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_panel(records):
    n = len(records)
    ref = np.load(_counts_path(REFERENCE_DT)).astype(np.float32)
    vmax = float(np.log1p(ref).max())

    fig, axes = plt.subplots(1, n, figsize=(min(3.2 * n, 44), 4.0))

    for ax, r in zip(axes, records):
        dt     = r["dt_ps"]
        counts = np.load(_counts_path(dt)).astype(np.float32)
        ax.imshow(np.log1p(counts), cmap="Blues", origin="lower",
                  extent=[-250, 250, -250, 250], aspect="equal",
                  vmin=0, vmax=vmax)
        if dt == REFERENCE_DT:
            subtitle = "(reference)"
        else:
            subtitle = f"corr = {r['image_correlation']:.4f}"
        ax.set_title(f"dt = {dt} ps\n{subtitle}", fontsize=9)
        ax.set_xlabel("y (mm)", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("z (mm)", fontsize=8)
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        f"Radiographs at each dt — z-pinch,  {N_PARTICLES:,} particles",
        fontsize=11,
    )
    fig.tight_layout()
    out = PLOTS / "convergence_dt_panel.png"
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

    if args.plot_only:
        print("Loading saved results from disk...")
        records = records_from_disk()
    else:
        print(f"Running {len(DT_PS_VALUES)} simulations  "
              f"(N = {N_PARTICLES:,},  total time = {TOTAL_TIME_PS:.0f} ps)...")
        records = run_all(verbose=args.verbose)

    print("Computing metrics vs reference...")
    records = compute_metrics(records)

    print(f"\n{'dt (ps)':>8}  {'max_steps':>10}  {'n_hits':>8}  {'hit_frac':>9}  "
          f"{'MSE_norm':>12}  {'corr':>10}  {'peak_rel':>9}  {'wall (s)':>9}")
    for r in records:
        wall = f"{r['runtime_s']:.2f}" if r["runtime_s"] is not None else "—"
        print(f"{r['dt_ps']:>8.2f}  {r['max_steps']:>10,}  {r['n_hits']:>8,}  "
              f"{r['hit_fraction']:>9.4f}  {r['mse_normalised']:>12.3e}  "
              f"{r['image_correlation']:>10.6f}  {r['peak_fluence_rel']:>9.4f}  {wall:>9}")

    out = RESULTS / "convergence_dt.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved → {out}")

    print("Generating plots...")
    plot_convergence(records)
    plot_panel(records)
    print("Done.")


if __name__ == "__main__":
    main()
