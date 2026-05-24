#!/usr/bin/env python3
"""
Regenerate all paper figures and benchmark plots.

Sections
--------
  figures      paper/figures/  paraxial_breakdown, energy_sweep, spectra_comparison
  convergence  docs/images/benchmark/  convergence_dt, _N, _grid, _grid_analytic
  benchmarks   docs/images/benchmark/  perf, plasmapy, physics_A-E

Usage
-----
  python3 scripts/reproduce_paper.py                       # full reproduction
  python3 scripts/reproduce_paper.py --fast                # sanity / CI
  python3 scripts/reproduce_paper.py --sections figures    # one section only
  python3 scripts/reproduce_paper.py --build               # cargo build first
  python3 scripts/reproduce_paper.py --fast --sections convergence,figures

Full vs fast
------------
  Full:  all simulations run from scratch at paper-quality particle counts.
  Fast:  - paper figures: 50k particles (vs 300-500k)
         - convergence:   --plot-only if cached .npy files exist; else full run
         - perf:          --quick (1e4, 1e5 only)
         - plasmapy:      same (already fast)
         - physics:       same (already fast per case)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).parent.parent.resolve()
SCRIPTS   = ROOT / "scripts"
BENCH     = ROOT / "benchmarks"
RESULTS   = BENCH / "results"

# ── colour codes ──────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _run(label, cmd, cwd=None):
    """Run a Python script (cmd[0]) with args cmd[1:], time it, return (ok, elapsed)."""
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(cmd[0])] + [str(a) for a in cmd[1:]],
        cwd=cwd or ROOT,
    )
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  {status}  {label:45s}  {elapsed:5.1f}s")
    return ok, elapsed


def _plot_only_ok(section):
    """Return True if cached .npy files exist for the given convergence section."""
    patterns = {
        "dt":             [RESULTS / f"counts_dt_{dt}ps.npy" for dt in ["0.05", "0.10", "0.20"]],
        "N":              [RESULTS / f"counts_N_{n}.npy" for n in [1_000_000, 500_000]],
        "grid":           [RESULTS / f"counts_grid_64x64x128.npy"],
        "grid_analytic":  [RESULTS / f"counts_gauss_256x256.npy"],
    }
    files = patterns.get(section, [])
    return all(f.exists() for f in files)


# ── sections ──────────────────────────────────────────────────────────────────

def section_figures(fast, results):
    """Regenerate paper/figures/ (paraxial_breakdown, energy_sweep, spectra_comparison)."""
    extra = ["--fast"] if fast else []
    for name, script in [
        ("paraxial_breakdown", SCRIPTS / "fig_paraxial_breakdown.py"),
        ("energy_sweep",       SCRIPTS / "fig_energy_sweep.py"),
        ("spectra_comparison", SCRIPTS / "fig_spectra.py"),
    ]:
        ok, t = _run(f"figures/{name}", [str(script)] + extra)
        results.append((f"figures/{name}", ok, t))


def section_convergence(fast, results):
    """Regenerate convergence plots (dt, N, grid, grid_analytic)."""
    studies = [
        ("convergence/dt",           BENCH / "run_convergence_dt.py",           "dt"),
        ("convergence/N",            BENCH / "run_convergence_N.py",            "N"),
        ("convergence/grid",         BENCH / "run_convergence_grid.py",         "grid"),
        ("convergence/grid_analytic",BENCH / "run_convergence_grid_analytic.py","grid_analytic"),
    ]
    for label, script, key in studies:
        if fast and _plot_only_ok(key):
            extra = ["--plot-only"]
        else:
            extra = []
        ok, t = _run(label, [str(script)] + extra)
        results.append((label, ok, t))


def section_benchmarks(fast, results):
    """Regenerate benchmark plots (perf, plasmapy, physics)."""
    # Physics cases
    ok, t = _run("benchmarks/physics", [str(BENCH / "run_physics.py")])
    results.append(("benchmarks/physics", ok, t))

    # Performance
    perf_flags = ["--quick"] if fast else []
    ok, t = _run("benchmarks/perf", [str(BENCH / "run_perf.py")] + perf_flags)
    results.append(("benchmarks/perf", ok, t))

    # PlasmaPy comparison (skip in fast if no plasmapy installed)
    try:
        import plasmapy  # noqa: F401
        ok, t = _run("benchmarks/plasmapy", [str(BENCH / "run_plasmapy.py")])
        results.append(("benchmarks/plasmapy", ok, t))
    except ImportError:
        print(f"  {YELLOW}SKIP{RESET}  benchmarks/plasmapy"
              f"                               (plasmapy not installed)")
        results.append(("benchmarks/plasmapy", None, 0.0))

    # Aggregate plots + regenerate docs/benchmark.md
    ok, t = _run("benchmarks/plot", [str(BENCH / "plot.py")])
    results.append(("benchmarks/plot", ok, t))


# ── main ──────────────────────────────────────────────────────────────────────

ALL_SECTIONS = ["figures", "convergence", "benchmarks"]

SECTION_FNS = {
    "figures":     section_figures,
    "convergence": section_convergence,
    "benchmarks":  section_benchmarks,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fast", action="store_true",
                    help="Reduced-N figures, --plot-only for cached convergence, --quick perf")
    ap.add_argument("--sections", default=",".join(ALL_SECTIONS),
                    help=f"Comma-separated subset of: {', '.join(ALL_SECTIONS)}")
    ap.add_argument("--build", action="store_true",
                    help="Run 'cargo build --release' before generating figures")
    args = ap.parse_args()

    sections = [s.strip() for s in args.sections.split(",")]
    unknown  = [s for s in sections if s not in ALL_SECTIONS]
    if unknown:
        ap.error(f"Unknown sections: {unknown}.  Valid: {ALL_SECTIONS}")

    if args.build:
        print(f"\n{BOLD}Building…{RESET}")
        r = subprocess.run(["cargo", "build", "--release"], cwd=ROOT / "rust")
        if r.returncode != 0:
            print(f"{RED}Build failed — aborting.{RESET}")
            sys.exit(1)

    mode = "fast" if args.fast else "full"
    print(f"\n{BOLD}reproduce_paper.py  [{mode}]  sections: {', '.join(sections)}{RESET}")
    print(f"{'─' * 65}")

    wall_start = time.time()
    results    = []

    for sec in sections:
        print(f"\n{BOLD}{sec.upper()}{RESET}")
        SECTION_FNS[sec](args.fast, results)

    # ── summary ───────────────────────────────────────────────────────────────
    total = time.time() - wall_start
    n_ok   = sum(1 for _, ok, _ in results if ok is True)
    n_fail = sum(1 for _, ok, _ in results if ok is False)
    n_skip = sum(1 for _, ok, _ in results if ok is None)

    print(f"\n{'─' * 65}")
    print(f"{BOLD}Summary{RESET}  {n_ok} passed  {n_fail} failed  {n_skip} skipped"
          f"  ({total:.1f}s total)\n")

    if n_fail:
        print(f"{RED}Failed steps:{RESET}")
        for label, ok, _ in results:
            if ok is False:
                print(f"  ✗  {label}")
        print()
        sys.exit(1)

    # List generated outputs
    paper_figs = sorted((ROOT / "paper" / "figures").glob("*.pdf"))
    bench_imgs = sorted((ROOT / "docs" / "images" / "benchmark").glob("*.png"))
    print(f"paper/figures/  ({len(paper_figs)} PDFs)")
    for p in paper_figs:
        print(f"  {p.name}")
    print(f"\ndocs/images/benchmark/  ({len(bench_imgs)} PNGs)")
    for p in bench_imgs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
