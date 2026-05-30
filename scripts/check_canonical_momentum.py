#!/usr/bin/env python3
"""
Check canonical angular momentum conservation from a diag-canonical CSV.

P_φ = m(x·uy − y·ux) + (q·Bz/2)(x²+y²)

For a proton in uniform Bz with E=0, P_φ is exactly conserved.
The Boris integrator is symplectic for pure magnetic fields, so the
relative drift in P_φ should stay at or below machine precision.

Usage:
    python3 scripts/check_canonical_momentum.py diag_canonical.csv [--bz 1.0]
    python3 scripts/check_canonical_momentum.py diag_canonical.csv --plot
"""

import argparse
import sys
import numpy as np
import pandas as pd

# Physical constants
M_P = 1.6726219e-27   # proton mass [kg]
Q_P = 1.6021766e-19   # proton charge [C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV from proton-tracer diag-canonical")
    ap.add_argument("--bz", type=float, default=1.0,
                    help="Bz field used in the run [T] (default: 1.0)")
    ap.add_argument("--plot", action="store_true",
                    help="Show matplotlib plot of P_φ drift")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    bz = args.bz

    # P_φ = m(x·uy − y·ux) + (q·Bz/2)(x²+y²)
    Lphi = M_P * (df["x_m"] * df["uy_ms"] - df["y_m"] * df["ux_ms"])
    mag  = (Q_P * bz / 2.0) * (df["x_m"]**2 + df["y_m"]**2)
    df["Pphi"] = Lphi + mag

    particles = sorted(df["particle"].unique())
    print(f"{'Particle':>8}  {'P_φ(0) [kg·m²/s]':>20}  {'max |ΔP_φ/P_φ(0)|':>20}  {'status':>8}")
    print("-" * 70)

    all_pass = True
    tol = 1e-4  # 0.01% — Boris in pure B conserves to float32 rounding

    for pid in particles:
        sub = df[df["particle"] == pid].sort_values("step")
        pphi = sub["Pphi"].values
        p0 = pphi[0]
        if abs(p0) < 1e-40:
            print(f"{pid:>8}  {'(degenerate)':>20}")
            continue
        rel_drift = np.max(np.abs((pphi - p0) / p0))
        ok = rel_drift < tol
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"{pid:>8}  {p0:>20.6e}  {rel_drift:>20.6e}  {status:>8}")

    print()
    if all_pass:
        print("RESULT: PASS — P_φ conserved to within tolerance for all particles.")
    else:
        print("RESULT: FAIL — P_φ drifted beyond tolerance for at least one particle.")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
            for pid in particles:
                sub = df[df["particle"] == pid].sort_values("step")
                pphi = sub["Pphi"].values
                p0 = pphi[0]
                if abs(p0) < 1e-40:
                    continue
                rel = (pphi - p0) / p0
                ax.plot(sub["step"].values, rel, label=f"particle {pid}")
            ax.set_xlabel("step")
            ax.set_ylabel(r"$(P_\varphi - P_{\varphi,0}) \;/\; P_{\varphi,0}$")
            ax.set_title(f"Canonical angular momentum conservation  (Bz = {bz} T)")
            ax.legend()
            ax.axhline(0, color="k", lw=0.5, ls="--")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available — skipping plot")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
