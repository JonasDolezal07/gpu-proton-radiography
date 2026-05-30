#!/usr/bin/env python3
"""
Fiducial mesh radiograph demo — prad opaque absorber mode.

Places a wire-mesh absorber in the beam path and runs two simulations:
  1. No field  — straight grid shadows (undeflected reference)
  2. Z-pinch   — grid distorted by the integrated B field

This reproduces the experimental technique where a physical mesh placed
upstream of the target imprints a reference grid on the detector; field-
induced deflections show up as distortions of that grid.

Usage
-----
    python3 scripts/demo_mesh_radiograph.py
    python3 scripts/demo_mesh_radiograph.py --n 500000 --output docs/images/mesh_radiograph.png
    python3 scripts/demo_mesh_radiograph.py --build   # cargo build first
"""

import argparse
import os
import struct
import subprocess
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT        = Path(__file__).resolve().parent.parent
BIN         = ROOT / "rust/target/release/proton_tracer"
ZPINCH_BFLD = ROOT / "data/instabilities/zpinch.bfld"
OUT_DIR     = ROOT / "scripts/output/mesh_demo"


# ── Vulkan env ────────────────────────────────────────────────────────────────

def vulkan_env():
    env = os.environ.copy()
    icd = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    lib = Path("/opt/homebrew/lib")
    if icd.exists() and "VK_ICD_FILENAMES" not in env:
        env["VK_ICD_FILENAMES"] = str(icd)
    if lib.exists():
        ex = env.get("DYLD_LIBRARY_PATH", "")
        if str(lib) not in ex:
            env["DYLD_LIBRARY_PATH"] = (str(lib) + ":" + ex).rstrip(":")
    return env


# ── Density file ──────────────────────────────────────────────────────────────

def write_dens(path, rho, bounds):
    nx, ny, nz = rho.shape
    xmn, xmx, ymn, ymx, zmn, zmx = bounds
    with open(path, "wb") as f:
        f.write(b"DENS")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<III", nx, ny, nz))
        f.write(struct.pack("<6f", xmn, xmx, ymn, ymx, zmn, zmx))
        f.write(b"\x00" * (64 - 4 - 4 - 12 - 24))
        f.write(rho.astype("<f4").tobytes())


def make_mesh_dens(path, pitch_mm=5, wire_mm=1, extent_mm=50):
    """
    Wire mesh in the y-z plane.

    Grid cells where (j % pitch) < wire OR (k % pitch) < wire
    are filled with opaque material (10 g/cm³ >> 0.1 threshold).
    Cell size = 1 mm. Mesh is 2 mm thick in x, placed at x = -60 to -58 mm
    (well upstream of the zpinch field region which starts at x ~ -50 mm).
    """
    n = int(extent_mm * 2)  # 1 mm cells over [-extent, +extent]
    rho = np.zeros((2, n, n), dtype=np.float32)
    for j in range(n):
        for k in range(n):
            if (j % pitch_mm) < wire_mm or (k % pitch_mm) < wire_mm:
                rho[:, j, k] = 10.0
    bounds = (
        -0.060, -0.058,
        -extent_mm * 1e-3, extent_mm * 1e-3,
        -extent_mm * 1e-3, extent_mm * 1e-3,
    )
    write_dens(path, rho, bounds)
    coverage = (rho[0] > 0).sum() / rho[0].size
    print(f"  Mesh: {n}×{n} cells, {pitch_mm} mm pitch, {wire_mm} mm wires, "
          f"{coverage*100:.0f}% coverage")


# ── TOML deck ─────────────────────────────────────────────────────────────────

def make_deck(field_path, dens_path, n_particles, scale_B):
    return f"""\
[field]
path = "{field_path}"
scale_B = {scale_B}
scale_E = 0.0

[source]
type = "parallel"
direction = [1.0, 0.0, 0.0]
beam_radius_mm = 40.0
source_distance_mm = 80.0
energy_MeV = 14.7
n_particles = {n_particles}

[detector]
center_mm = [100.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [512, 512]

[density]
path = "{dens_path}"
material = "Al"
mode = "opaque"
opaque_threshold_g_cm3 = 0.1
"""


# ── Simulation ────────────────────────────────────────────────────────────────

def run_sim(deck_path, out_dir):
    env = vulkan_env()
    r = subprocess.run(
        [str(BIN), "run", str(deck_path), "-o", str(out_dir), "--overwrite"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if r.returncode != 0:
        print(r.stderr[-600:])
        raise RuntimeError(f"Simulation failed: {deck_path.name}")
    absorbed = None
    for line in (r.stdout + r.stderr).splitlines():
        if "absorbed" in line.lower():
            absorbed = line.strip()
    return absorbed


def read_counts(run_dir):
    data = (run_dir / "counts" / "processed_counts.bin").read_bytes()
    arr = np.frombuffer(data, dtype="<f4")
    n = int(len(arr) ** 0.5)
    return arr.reshape(n, n)


# ── Plot ──────────────────────────────────────────────────────────────────────

def gamma(img, g=0.45):
    m = img.max()
    return (img / m) ** g if m > 0 else img


def render_side_by_side(img_ref, img_field, out_path):
    # Crop to beam region. Detector: 500 mm, image: N×N px → px_per_mm = N/500.
    N = img_ref.shape[0]
    px_per_mm = N / 500.0
    cx = cy = N // 2
    half = int(90 * px_per_mm)  # ±90 mm
    mm_half = half / px_per_mm
    sl = (slice(cx - half, cx + half), slice(cy - half, cy + half))
    r = gamma(img_ref[sl])
    f = gamma(img_field[sl])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    kw = dict(cmap="hot", vmin=0, vmax=1, origin="lower",
              extent=[-mm_half, mm_half, -mm_half, mm_half])

    axes[0].imshow(r, **kw)
    axes[0].set_title("No field  —  undeflected mesh", fontsize=13)
    axes[0].set_xlabel("y  (mm)"); axes[0].set_ylabel("z  (mm)")

    axes[1].imshow(f, **kw)
    axes[1].set_title("Z-pinch field  —  mesh deflected by B", fontsize=13)
    axes[1].set_xlabel("y  (mm)")

    fig.suptitle("Fiducial mesh radiograph: opaque absorber mode\n"
                 "5 mm pitch · 1 mm wire · 14.7 MeV parallel beam",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000, help="particles per run")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "scripts/output/mesh_radiograph.png")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    if args.build:
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT / "rust", check=True)

    if not BIN.exists():
        print("Binary not found — run with --build or: cd rust && cargo build --release")
        sys.exit(1)

    if not ZPINCH_BFLD.exists():
        print(f"Z-pinch field not found: {ZPINCH_BFLD}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dens_path = OUT_DIR / "mesh.dens"
    deck_ref   = OUT_DIR / "deck_nofield.toml"
    deck_field = OUT_DIR / "deck_zpinch.toml"
    run_ref    = OUT_DIR / "run_nofield"
    run_field  = OUT_DIR / "run_zpinch"

    print("Generating mesh density file...")
    make_mesh_dens(dens_path)

    deck_ref.write_text(make_deck(ZPINCH_BFLD, dens_path, args.n, scale_B=0.0))
    deck_field.write_text(make_deck(ZPINCH_BFLD, dens_path, args.n, scale_B=1.0))

    print(f"\nRun 1/2: no field  (N={args.n:,})")
    ab = run_sim(deck_ref, run_ref)
    if ab: print(f"  {ab}")

    print(f"Run 2/2: z-pinch   (N={args.n:,})")
    ab = run_sim(deck_field, run_field)
    if ab: print(f"  {ab}")

    print("\nRendering comparison...")
    img_ref   = read_counts(run_ref)
    img_field = read_counts(run_field)
    render_side_by_side(img_ref, img_field, args.output)


if __name__ == "__main__":
    main()
