#!/usr/bin/env python3
"""
Generate benchmark plots and docs/benchmark.md.

Reads benchmarks/results/perf_results.json and physics_results.json.
Writes benchmarks/plots/*.png and docs/benchmark.md.

Usage:
    python3 benchmarks/plot.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
import utils

RESULTS = utils.RESULTS_DIR
PLOTS   = utils.PLOTS_DIR
DOCS    = utils.ROOT / "docs"

FIELD_COLORS = {
    "zero field": "#5b9bd5",
    "z-pinch":    "#e05c5c",
    "kink":       "#f0a830",
    "sausage":    "#5dbf5d",
}
CMAP = "Blues"


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ── Performance plots ─────────────────────────────────────────────────────────

def plot_perf(records):
    by_field = defaultdict(list)
    for r in records:
        by_field[r["field_label"]].append(r)
    for recs in by_field.values():
        recs.sort(key=lambda x: x["n_particles"])

    gpu = records[0].get("gpu", "GPU") if records else "GPU"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Runtime
    ax = axes[0]
    for lbl, recs in by_field.items():
        ns = [r["n_particles"]  for r in recs]
        ts = [r["runtime_s"]    for r in recs]
        ax.plot(ns, ts, "o-", color=FIELD_COLORS.get(lbl, "#888"),
                label=lbl, lw=2, ms=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Particle count", fontsize=12)
    ax.set_ylabel("Wall time (s)", fontsize=12)
    ax.set_title("Runtime vs particle count", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.text(0.98, 0.04, gpu, transform=ax.transAxes,
            fontsize=8, ha="right", va="bottom", color="#666")

    # Throughput
    ax = axes[1]
    for lbl, recs in by_field.items():
        ns = [r["n_particles"]            for r in recs]
        tp = [r["particles_per_s"] / 1e6 for r in recs]
        ax.plot(ns, tp, "o-", color=FIELD_COLORS.get(lbl, "#888"),
                label=lbl, lw=2, ms=7)
    ax.set_xscale("log")
    ax.set_xlabel("Particle count", fontsize=12)
    ax.set_ylabel("Throughput (Mparticles / s)", fontsize=12)
    ax.set_title("GPU throughput vs particle count", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = PLOTS / "perf.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  {out.name}")


# ── Physics plots ─────────────────────────────────────────────────────────────

def plot_case_A(res):
    counts_path = RESULTS / "counts_A.npy"
    if not counts_path.exists():
        print("  case A: counts_A.npy not found, skipping")
        return
    counts = np.load(counts_path)
    my, mz = res["mean_y_mm"], res["mean_z_mm"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.log1p(counts), cmap=CMAP, origin="lower",
              extent=[-250, 250, -250, 250], aspect="equal")
    ax.set_xlabel("y (mm)", fontsize=11)
    ax.set_ylabel("z (mm)", fontsize=11)
    ax.set_title("Case A: zero field — hit distribution\n(should be uniform disk)", fontsize=11)
    ax.text(0.97, 0.03,
            f"mean y = {my:+.2f} mm\nmean z = {mz:+.2f} mm",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.tight_layout()
    out = PLOTS / "physics_A.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  {out.name}")


def plot_case_B(res):
    recs    = res["records"]
    B       = [r["B_T"]                        for r in recs]
    meas    = [r["measured_mm"]                for r in recs]
    exact   = [r["predicted_mm"]               for r in recs]
    parax   = [r.get("paraxial_mm", r["predicted_mm"]) for r in recs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(B, parax, "k--", lw=1.5, label="Paraxial (small-angle)")
    ax.plot(B, exact, "b-",  lw=1.5, label="Exact circular arc")
    ax.plot(B, meas,  "ro",  ms=9,   label="Boris (GPU)")
    ax.set_xlabel("Bz (T)", fontsize=12)
    ax.set_ylabel("Mean y-displacement at detector (mm)", fontsize=11)
    ax.set_title("Case B: uniform Bz — cyclotron deflection", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Relative error vs exact circular arc
    ax = axes[1]
    errs = [r["relative_error"] for r in recs if r["relative_error"] is not None]
    Bs   = [r["B_T"]            for r in recs if r["relative_error"] is not None]
    ax.bar(Bs, [e * 100 for e in errs], width=0.08,
           color=["#e05c5c" if abs(e) > 0.05 else "#5dbf5d" for e in errs])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Bz (T)", fontsize=12)
    ax.set_ylabel("Relative error (%)", fontsize=11)
    ax.set_title("Boris vs exact circular arc: relative error", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = PLOTS / "physics_B.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  {out.name}")


def plot_case_CD(res_C, res_D):
    for res, tag in [(res_C, "C"), (res_D, "D")]:
        if res is None:
            continue
        boris_path    = RESULTS / f"counts_boris_{tag}.npy"
        paraxial_path = RESULTS / f"counts_paraxial_{tag}.npy"
        if not boris_path.exists():
            print(f"  case {tag}: numpy arrays not found, skipping")
            continue

        boris    = np.load(boris_path)
        paraxial = np.load(paraxial_path)
        corr     = res["correlation"]
        label    = res["label"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, img, title in zip(
            axes,
            [paraxial, boris],
            ["Paraxial approximation\n(straight-line integral)", "Boris integrator\n(full GPU orbit)"],
        ):
            ax.imshow(np.log1p(img), cmap=CMAP, origin="lower",
                      extent=[-250, 250, -250, 250], aspect="equal")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("y (mm)", fontsize=10)
            ax.set_ylabel("z (mm)", fontsize=10)

        fig.suptitle(
            f"Case {tag}: z-pinch {label}\nImage correlation: {corr:.3f}",
            fontsize=12, y=1.01,
        )
        fig.tight_layout()
        out = PLOTS / f"physics_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {out.name}")


# ── docs/benchmark.md ────────────────────────────────────────────────────────

def write_md(perf, phys):
    gpu = perf[0].get("gpu", "GPU") if perf else "GPU"

    by_field = defaultdict(dict)
    for r in perf:
        by_field[r["field_label"]][r["n_particles"]] = r
    ns_all     = sorted({r["n_particles"] for r in perf})
    field_lbls = list(dict.fromkeys(r["field_label"] for r in perf))

    case_A = phys.get("case_A", {})
    case_B = phys.get("case_B", {})
    case_C = phys.get("case_C")
    case_D = phys.get("case_D")

    L = []

    L += [
        "# Benchmark",
        "",
        "> Generated by `python3 benchmarks/plot.py`.",
        "> Re-run `benchmarks/run_perf.py` and `benchmarks/run_physics.py` to refresh data.",
        "",
        "---",
        "",
        "## 1. Validation cases",
        "",
        "These cases verify that the Boris integrator agrees with analytical predictions",
        "in regimes where those predictions are known to be accurate.",
        "",
    ]

    # Case A
    L += ["### Case A — Zero field: straight-line projection", ""]
    L += ["With no electromagnetic field, protons travel in straight lines.",
          "All hits should cluster within 2 mm of the detector centre.", ""]
    if case_A:
        ok = "**pass**" if case_A.get("centring_ok") else "**FAIL**"
        L += [
            "| Metric | Value |",
            "|---|---|",
            f"| Mean y | {case_A['mean_y_mm']:+.2f} mm |",
            f"| Mean z | {case_A['mean_z_mm']:+.2f} mm |",
            f"| Std y  | {case_A['std_y_mm']:.1f} mm |",
            f"| Std z  | {case_A['std_z_mm']:.1f} mm |",
            f"| Centring (< 2 mm) | {ok} |",
            "",
            "![Case A — zero field hit distribution](../benchmarks/plots/physics_A.png)",
            "",
        ]

    # Case B
    L += ["### Case B — Uniform Bz: cyclotron deflection", ""]
    L += [
        "Protons crossing a uniform transverse Bz field through a finite slab undergo a",
        "lateral displacement. The exact prediction (circular arc inside slab + straight drift after) is:",
        "",
        "```",
        "R      = p / (q Bz)                    cyclotron radius",
        "φ      = arcsin(L / R)                 arc angle through slab",
        "Δy     = −R(1 − cos φ) − tan(φ) × lever_arm",
        "```",
        "",
        "where `L = 0.10 m` is the field slab length and `lever_arm = 0.05 m` is the",
        "distance from field exit to detector.  The small-angle (paraxial) limit is also shown.",
        "",
    ]
    if case_B:
        L += ["| Bz (T) | Boris (mm) | Exact circ arc (mm) | Paraxial (mm) | Err vs exact |",
              "|---|---|---|---|---|"]
        for r in case_B.get("records", []):
            err   = f"{r['relative_error']:+.1%}"            if r.get("relative_error")           is not None else "—"
            parax = f"{r['paraxial_mm']:+.2f}"               if r.get("paraxial_mm")              is not None else "—"
            L.append(f"| {r['B_T']:.1f} | {r['measured_mm']:+.2f} | {r['predicted_mm']:+.2f} | {parax} | {err} |")
        L += ["", "![Case B — cyclotron deflection](../benchmarks/plots/physics_B.png)", ""]

    L += [
        "---",
        "",
        "## 2. Paraxial breakdown study",
        "",
        "The paraxial approximation integrates B⊥ along the *unperturbed* straight-line",
        "path to predict hit positions. It is accurate when deflections are small.",
        "In strong, structured fields it fails: trajectories curve significantly,",
        "caustics form, and the full-orbit result differs markedly from the prediction.",
        "",
    ]

    # Case C
    L += ["### Case C — Weak z-pinch (scale\\_B = 0.1×)", ""]
    L += ["At 10% of nominal field strength deflections are reduced, but the z-pinch is still a",
          "strong field: 95th-percentile |θ| ≈ 10°, so paraxial agreement is only moderate.", ""]
    if case_C:
        L += [
            f"**Image correlation Boris vs paraxial: {case_C['correlation']:.3f}**",
            "",
            f"95th-percentile deflection angle |θ|: {case_C['theta_95th_deg']:.2f}°",
            "",
            "![Case C — weak z-pinch](../benchmarks/plots/physics_C.png)",
            "",
        ]

    # Case D
    L += ["### Case D — Strong z-pinch (scale\\_B = 1.0×)", ""]
    L += ["At full field strength, particle paths curve substantially. The paraxial approximation",
          "breaks down: particles predicted to pass through the centre instead form caustic",
          "arcs visible in the full-orbit radiograph.", ""]
    if case_D:
        L += [
            f"**Image correlation Boris vs paraxial: {case_D['correlation']:.3f}**",
            "",
            f"95th-percentile deflection angle |θ|: {case_D['theta_95th_deg']:.2f}°",
            "",
            "![Case D — strong z-pinch](../benchmarks/plots/physics_D.png)",
            "",
        ]

    L += [
        "---",
        "",
        "## 3. Performance scaling",
        "",
        f"Hardware: **{gpu}**",
        "",
        "![Performance plots](../benchmarks/plots/perf.png)",
        "",
        "### Timing table",
        "",
    ]
    header = "| Field | " + " | ".join(f"{n:,}" for n in ns_all) + " |"
    sep    = "|---|" + "---|" * len(ns_all)
    L += [header, sep]
    for lbl in field_lbls:
        row = f"| {lbl} |"
        for n in ns_all:
            r = by_field[lbl].get(n)
            row += f" {r['runtime_s']:.2f} s |" if r else " — |"
        L.append(row)
    L += ["", "Wall time in seconds.", ""]

    out = DOCS / "benchmark.md"
    out.write_text("\n".join(L))
    print(f"  docs/benchmark.md")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    PLOTS.mkdir(parents=True, exist_ok=True)

    perf = load(RESULTS / "perf_results.json") or []
    phys = load(RESULTS / "physics_results.json") or {}

    print("Performance plots:")
    if perf:
        plot_perf(perf)
    else:
        print("  perf_results.json not found — run run_perf.py first")

    print("Physics plots:")
    if phys:
        if "case_A" in phys: plot_case_A(phys["case_A"])
        if "case_B" in phys: plot_case_B(phys["case_B"])
        plot_case_CD(phys.get("case_C"), phys.get("case_D"))
    else:
        print("  physics_results.json not found — run run_physics.py first")

    print("Writing benchmark.md:")
    write_md(perf, phys)

    print("Done.")


if __name__ == "__main__":
    main()
