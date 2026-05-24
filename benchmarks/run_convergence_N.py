#!/usr/bin/env python3
"""
Particle-count convergence study.

Runs the standard z-pinch benchmark at seven particle counts spanning three
decades (1k – 1M) with fixed timestep and geometry. All results are compared
against the 1M-particle reference using normalised MSE and image correlation.

Expected behaviour: MSE ~ 1/N (Poisson shot noise) on a log-log plot with
slope -1, levelling to a noise floor as N approaches the reference count.
Correlation rises toward 1.0 with increasing N.

Results:  benchmarks/results/convergence_N.json
Plots:    docs/images/benchmark/convergence_N.png
          docs/images/benchmark/convergence_N_panel.png

Usage:
    python3 benchmarks/run_convergence_N.py
    python3 benchmarks/run_convergence_N.py --verbose
    python3 benchmarks/run_convergence_N.py --plot-only
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

N_VALUES    = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
REFERENCE_N = 1_000_000
DT_PS       = 0.2
MAX_STEPS   = 25_000


def _run_dir(n):
    return RESULTS / "runs" / f"conv_N_{n}"


def _counts_path(n):
    return RESULTS / f"counts_N_{n}.npy"


# ── simulation ────────────────────────────────────────────────────────────────

def run_all(verbose=False):
    utils.setup_dirs()
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)

    records = []
    for n in N_VALUES:
        out = _run_dir(n)
        print(f"  N = {n:>9,}")

        meta = utils.run_tracer(
            DECK, out,
            overrides={
                "source.n_particles": str(n),
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
            n_particles=n,
            n_hits=hits,
            hit_fraction=meta["diagnostics"]["hit_fraction"],
            peak_counts=float(counts.max()),
            runtime_s=wall,
        ))

    return records


def records_from_disk():
    records = []
    for n in N_VALUES:
        npy = _counts_path(n)
        if not npy.exists():
            print(f"Missing {npy} — run without --plot-only first")
            sys.exit(1)
        counts = np.load(npy).astype(np.float32)
        meta_path = _run_dir(n) / "metadata.json"
        wall, hits, hit_frac = None, int(counts.sum()), float(counts.sum()) / n
        if meta_path.exists():
            with open(meta_path) as f:
                m = json.load(f)
            wall     = m["performance"]["total_runtime_s"]
            hits     = m["diagnostics"]["n_hits"]
            hit_frac = m["diagnostics"]["hit_fraction"]
        records.append(dict(
            n_particles=n, n_hits=hits, hit_fraction=hit_frac,
            peak_counts=float(counts.max()), runtime_s=wall,
        ))
    return records


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(records):
    ref = np.load(_counts_path(REFERENCE_N)).astype(np.float64)
    ref_norm = ref / ref.sum()
    ref_peak = ref.max()

    for r in records:
        img  = np.load(_counts_path(r["n_particles"])).astype(np.float64)
        norm = img / img.sum() if img.sum() > 0 else img

        mse      = float(np.mean((ref_norm - norm) ** 2))
        corr     = float(np.corrcoef(ref_norm.ravel(), norm.ravel())[0, 1])
        peak_rel = float(norm.max() / ref_norm.max()) if ref_norm.max() > 0 else 1.0

        r["mse_normalised"]    = mse
        r["image_correlation"] = corr
        r["peak_fluence_rel"]  = peak_rel

    return records


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_convergence(records):
    PLOTS.mkdir(parents=True, exist_ok=True)

    ns   = [r["n_particles"]        for r in records]
    mse  = [r["mse_normalised"]     for r in records]
    corr = [r["image_correlation"]  for r in records]

    # Fit the 1/N theoretical line through the coarsest non-reference points.
    # MSE_theory = C/N; estimate C from the lowest-N points where ref noise is negligible.
    fit_ns  = np.array([r["n_particles"] for r in records if r["n_particles"] <= 50_000],
                       dtype=float)
    fit_mse = np.array([r["mse_normalised"] for r in records if r["n_particles"] <= 50_000])
    C_fit   = float(np.median(fit_mse * fit_ns))   # robust estimate of C
    theory_ns  = np.logspace(np.log10(min(ns)), np.log10(REFERENCE_N), 200)
    theory_mse = C_fit / theory_ns

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MSE vs N
    ax = axes[0]
    ax.loglog(ns, mse, "o-", color="#e05c5c", lw=2, ms=8, label="measured MSE", zorder=3)
    ax.loglog(theory_ns, theory_mse, "--", color="#aaa", lw=1.5,
              label="1/N  shot-noise scaling")
    ax.axvline(REFERENCE_N, color="#5b9bd5", ls=":", lw=1,
               label=f"reference  N = {REFERENCE_N:,}")
    ax.set_xlabel("N particles", fontsize=12)
    ax.set_ylabel("Normalised MSE vs 1M-particle reference", fontsize=11)
    ax.set_title("Particle-count convergence — MSE", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Correlation vs N
    ax = axes[1]
    ax.semilogx(ns, corr, "o-", color="#5b9bd5", lw=2, ms=8)
    ax.axvline(REFERENCE_N, color="#5b9bd5", ls=":", lw=1,
               label=f"reference  N = {REFERENCE_N:,}")
    ax.axhline(1.0, color="#ccc", ls=":", lw=1)
    ax.set_xlabel("N particles", fontsize=12)
    ax.set_ylabel("Image correlation with 1M-particle reference", fontsize=11)
    ax.set_title("Particle-count convergence — correlation", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Particle-count convergence — z-pinch field,  "
        f"dt = {DT_PS} ps,  max_steps = {MAX_STEPS:,}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / "convergence_N.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_panel(records):
    n_panels = len(records)
    fig, axes = plt.subplots(1, n_panels, figsize=(min(3.2 * n_panels, 44), 4.0))

    ref = np.load(_counts_path(REFERENCE_N)).astype(np.float32)
    vmax = float(np.log1p(ref).max())

    for ax, r in zip(axes, records):
        n      = r["n_particles"]
        counts = np.load(_counts_path(n)).astype(np.float32)
        ax.imshow(np.log1p(counts), cmap="Blues", origin="lower",
                  extent=[-250, 250, -250, 250], aspect="equal",
                  vmin=0, vmax=vmax)
        if n == REFERENCE_N:
            subtitle = "(reference)"
        else:
            subtitle = f"corr = {r['image_correlation']:.3f}"
        label = f"N = {n:,}\n{subtitle}"
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("y (mm)", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("z (mm)", fontsize=7)
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        f"Radiographs at each particle count — z-pinch,  dt = {DT_PS} ps",
        fontsize=11,
    )
    fig.tight_layout()
    out = PLOTS / "convergence_N_panel.png"
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
        print(f"Running {len(N_VALUES)} simulations  "
              f"(dt = {DT_PS} ps,  max_steps = {MAX_STEPS:,})...")
        records = run_all(verbose=args.verbose)

    print("Computing metrics vs reference...")
    records = compute_metrics(records)

    print(f"\n{'N':>10}  {'n_hits':>10}  {'hit_frac':>9}  "
          f"{'MSE_norm':>12}  {'corr':>10}  {'peak_rel':>9}  {'wall (s)':>9}")
    for r in records:
        wall = f"{r['runtime_s']:.2f}" if r["runtime_s"] is not None else "—"
        print(f"{r['n_particles']:>10,}  {r['n_hits']:>10,}  {r['hit_fraction']:>9.4f}  "
              f"{r['mse_normalised']:>12.3e}  {r['image_correlation']:>10.6f}  "
              f"{r['peak_fluence_rel']:>9.4f}  {wall:>9}")

    out = RESULTS / "convergence_N.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved → {out}")

    print("Generating plots...")
    plot_convergence(records)
    plot_panel(records)
    print("Done.")


if __name__ == "__main__":
    main()
