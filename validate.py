#!/usr/bin/env python3
"""
Validation suite for the proton tracer E-field implementation.

Run from the project root:
    python3 validate.py          # uses existing binary
    python3 validate.py --build  # cargo build --release first

Tests:
    1  B-only regression  — zpinch preset produces expected hits
    2  Zero fields        — straight-line projection, mean deflection ≈ 0
    3  Uniform E only     — positive Ey deflects proton in +y (sign + magnitude)
    4  Uniform B only     — Boris conserves kinetic energy exactly (E = 0 case)
"""

import sys
import os
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).parent.resolve()
BIN     = ROOT / "rust/target/release/proton_tracer"
VALDATA = ROOT / "data/validation"
VALOUT  = ROOT / "output/validation"

PROTON_QM = 9.58e7   # q/m for proton [C/kg]
PROTON_V  = 5.27e7   # speed of a 14.7 MeV proton [m/s]


# ── helpers ──────────────────────────────────────────────────────────────────

def build():
    print("Building proton_tracer (release) …")
    r = subprocess.run(["cargo", "build", "--release"], cwd=ROOT / "rust")
    if r.returncode != 0:
        sys.exit("Build failed")
    print("Build OK\n")


def write_bfld(path, B, E, bounds):
    """
    Write a .bfld file.  B and E must be ndarray (nx,ny,nz,3) float32.
    Writes version 2 if E is nonzero anywhere, otherwise version 1.
    Data is stored C-contiguous (x outermost, z innermost, components last).
    """
    nx, ny, nz = B.shape[:3]
    has_e = E is not None and np.any(E != 0)
    version = 2 if has_e else 1
    xmn, xmx, ymn, ymx, zmn, zmx = bounds
    with open(path, "wb") as f:
        f.write(b"BFLD")
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<III", nx, ny, nz))
        f.write(struct.pack("<6f", xmn, xmx, ymn, ymx, zmn, zmx))
        f.write(b"\x00" * (64 - 4 - 4 - 12 - 24))   # padding to 64 bytes
        f.write(B.astype("<f4").tobytes())
        if has_e:
            f.write(E.astype("<f4").tobytes())


def write_config(path, field_file, *,
                 n_particles=50_000, energy_MeV=14.7,
                 beam_center=(-0.1, 0.0, 0.0),
                 beam_direction=(1.0, 0.0, 0.0),
                 beam_radius_mm=30.0,
                 detector_center_mm=(110.0, 0.0, 0.0),
                 detector_normal=(1.0, 0.0, 0.0),
                 detector_up=(0.0, 1.0, 0.0),
                 detector_width_mm=500.0,
                 detector_height_mm=500.0,
                 detector_pixels=(512, 512),
                 dt_ps=1.0):
    """Write a v2-format config.  Coordinate convention: +x is beam axis."""
    cfg = {
        "field_path": field_file,
        "detector": {
            "center_mm": list(detector_center_mm),
            "normal":    list(detector_normal),
            "up":        list(detector_up),
            "width_mm":  detector_width_mm,
            "height_mm": detector_height_mm,
            "pixels":    list(detector_pixels),
        },
        "source": {
            "source_type":   "parallel",
            "n_particles":   n_particles,
            "energy_MeV":    energy_MeV,
            "beam_center":   list(beam_center),
            "beam_direction": list(beam_direction),
            "beam_radius_mm": beam_radius_mm,
            "angular_spread_deg": 0.0,
        },
        "dt_ps":     dt_ps,
        "max_steps": 20_000,
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def run_batch(config_path, out_dir):
    # Wipe stale output so we never silently read a previous run's CSV.
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the Vulkan loader and MoltenVK ICD are visible to the subprocess
    # even when the shell profile has not been sourced (IDE / agent context).
    env = os.environ.copy()
    brew_lib = Path("/opt/homebrew/lib")
    icd      = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    if icd.exists() and "VK_ICD_FILENAMES" not in env:
        env["VK_ICD_FILENAMES"] = str(icd)
    if brew_lib.exists():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        if str(brew_lib) not in existing:
            env["DYLD_LIBRARY_PATH"] = (str(brew_lib) + ":" + existing).rstrip(":")

    result = subprocess.run(
        [str(BIN), str(config_path), "--batch", "-o", str(out_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    for line in (result.stdout + result.stderr).splitlines():
        upper = line.upper()
        if any(tok in upper for tok in ("ERROR", "WARN", "COMPLETE", "EXPORTED", "HITS")):
            print("   ", line.strip())
    return result.returncode == 0


def read_hits(out_dir):
    """Return list of (y_mm, z_mm, ke_MeV) from the first CSV in out_dir."""
    csvs = sorted(out_dir.glob("*.csv"))
    if not csvs:
        return []
    rows = []
    with open(csvs[0]) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("y"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
    return rows


def _mean(vals):
    return sum(vals) / len(vals)


def _std(vals, mean=None):
    if mean is None:
        mean = _mean(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check_png_output(out_dir):
    """Assert PNG, sidecar, raw_counts.bin, and processed_counts.bin were produced.
    Returns (pass: bool, info: dict) for inclusion in REPORT.
    """
    # PNG and sidecar land directly in out_dir (no png/ subdir)
    pngs  = sorted(out_dir.glob("radiograph_*.png"))
    metas = sorted(out_dir.glob("radiograph_*_meta.json"))

    raw_bins  = sorted(out_dir.glob("radiograph_*_raw_counts.bin"))
    proc_bins = sorted(out_dir.glob("radiograph_*_processed_counts.bin"))

    # Renderer always uses DETECTOR_RESOLUTION = 1024
    expected_count_bytes = 1024 * 1024 * 4  # 1024×1024 × 4 bytes

    png_ok   = len(pngs)      > 0 and pngs[0].stat().st_size > 0
    meta_ok  = len(metas)     > 0 and metas[0].stat().st_size > 0
    raw_ok   = len(raw_bins)  > 0 and raw_bins[0].stat().st_size  == expected_count_bytes
    proc_ok  = len(proc_bins) > 0 and proc_bins[0].stat().st_size == expected_count_bytes

    for desc, ok in [("PNG", png_ok), ("PNG sidecar", meta_ok),
                     ("raw_counts.bin", raw_ok), ("processed_counts.bin", proc_ok)]:
        if not ok:
            print(f"   {desc} MISSING or wrong size in {out_dir}")

    info = {
        "png_path":          str(pngs[0])      if pngs      else None,
        "meta_path":         str(metas[0])     if metas     else None,
        "raw_counts_path":   str(raw_bins[0])  if raw_bins  else None,
        "proc_counts_path":  str(proc_bins[0]) if proc_bins else None,
        "png_ok":            png_ok,
        "meta_ok":           meta_ok,
        "raw_counts_ok":     raw_ok,
        "proc_counts_ok":    proc_ok,
    }
    return (png_ok and meta_ok and raw_ok and proc_ok), info

REPORT = {}   # populated by each test, written to validation_report.json at end


# ── test 1: B-only regression ────────────────────────────────────────────────

def test1_regression():
    """zpinch preset must produce hits — proves the existing B-only path still works."""
    print("Test 1: B-only regression  (zpinch)")
    out = VALOUT / "t1_regression"
    cfg = ROOT / "data/instabilities/zpinch.json"
    if not cfg.exists():
        print(f"   SKIP: {cfg} not found")
        return None

    if not run_batch(cfg, out):
        print("   tracer returned non-zero exit")
        return False

    hits = read_hits(out)
    n = len(hits)
    print(f"   hits = {n}")
    ok = n >= 10_000
    if not ok:
        print(f"   expected ≥10 000 hits, got {n}")
    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test1_b_only_regression"] = {
        "pass": ok,
        "hits": n,
        "threshold": 10_000,
        **png_info,
    }
    return ok


# ── test 2: zero fields ───────────────────────────────────────────────────────

def test2_zero_fields():
    """B = E = 0: protons travel straight, mean deflection must be < 1 mm."""
    print("Test 2: Zero fields  (straight-line projection)")
    VALDATA.mkdir(parents=True, exist_ok=True)
    nx = ny = nz = 16
    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)
    B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    E = np.zeros_like(B)
    write_bfld(VALDATA / "t2_zero.bfld", B, E, bounds)
    # Field x_max = 0.06 m; detector 50 mm downstream → center at x = 110 mm
    write_config(VALDATA / "t2_zero.json", "t2_zero.bfld",
                 n_particles=50_000, detector_center_mm=(110.0, 0.0, 0.0))
    out = VALOUT / "t2_zero"

    if not run_batch(VALDATA / "t2_zero.json", out):
        print("   tracer error")
        return False

    hits = read_hits(out)
    if not hits:
        print("   no hits recorded")
        return False

    # y_mm, z_mm columns (detector-plane axes; both should be near zero)
    ys = [h[0] for h in hits]
    zs = [h[1] for h in hits]
    my, mz = _mean(ys), _mean(zs)
    tol = 1.0   # 1 mm
    ok = abs(my) <= tol and abs(mz) <= tol
    print(f"   hits = {len(hits)},  mean_y = {my:+.3f} mm,  mean_z = {mz:+.3f} mm")
    if not ok:
        print(f"   deflection exceeds ±{tol:.0f} mm tolerance")
    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test2_zero_fields"] = {
        "pass": ok,
        "mean_y_mm": round(my, 4),
        "mean_z_mm": round(mz, 4),
        "tolerance_mm": tol,
        **png_info,
    }
    return ok


# ── test 3: uniform E only ────────────────────────────────────────────────────

def test3_uniform_E():
    """
    B = 0, Ey = +10 MV/m everywhere.
    Protons (positive charge) must deflect in +y.
    Magnitude must be within a factor of 3 of the non-relativistic estimate.
    """
    print("Test 3: Uniform E only  (parabolic deflection, sign + magnitude)")
    VALDATA.mkdir(parents=True, exist_ok=True)
    nx = ny = nz = 16
    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)
    B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    E = np.zeros_like(B)
    E_y = 1e7    # V/m  (+y direction)
    E[..., 1] = E_y
    write_bfld(VALDATA / "t3_E.bfld", B, E, bounds)
    write_config(VALDATA / "t3_E.json", "t3_E.bfld",
                 n_particles=50_000, detector_center_mm=(110.0, 0.0, 0.0))
    out = VALOUT / "t3_E"

    if not run_batch(VALDATA / "t3_E.json", out):
        print("   tracer error")
        return False

    hits = read_hits(out)
    if not hits:
        print("   no hits recorded")
        return False

    # h[0] = y_mm (global y, aligned with detector y-axis = [0,1,0])
    # E_y deflects protons in +y, so deflection shows up in h[0]
    ys = [h[0] for h in hits]
    mean_y_mm = _mean(ys)

    # Non-relativistic analytic estimate (result in mm).
    # Source at x=-0.10 m, detector at x=0.06+0.05=0.11 m → Δx=0.21 m.
    t_total      = 0.21 / PROTON_V
    dy_approx_mm = 0.5 * PROTON_QM * E_y * t_total ** 2 * 1e3

    rel_err = abs(mean_y_mm - dy_approx_mm) / dy_approx_mm if dy_approx_mm else float("inf")
    lo, hi = dy_approx_mm / 3.0, dy_approx_mm * 3.0
    ok = mean_y_mm > 0 and lo < mean_y_mm < hi

    print(f"   hits = {len(hits)},  mean_y = {mean_y_mm:+.3f} mm"
          f"  (analytic ≈ {dy_approx_mm:+.3f} mm,  rel err = {rel_err:.4f})")

    if not ok:
        if mean_y_mm <= 0:
            print(f"   FAIL: wrong sign (Ey > 0 should give mean_y > 0)")
        else:
            print(f"   FAIL: magnitude {mean_y_mm:.3f} mm outside [{lo:.3f}, {hi:.3f}] mm")

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test3_uniform_E"] = {
        "pass": ok,
        "measured_mean_y_mm": round(mean_y_mm, 4),
        "analytic_mean_y_mm": round(dy_approx_mm, 4),
        "relative_error": round(rel_err, 6),
        "sign_correct": mean_y_mm > 0,
        **png_info,
    }
    return ok


# ── test 4: uniform B only, energy conservation ───────────────────────────────

def test4_B_energy_conservation():
    """
    E = 0, Bz = 1 T everywhere.
    Boris is exactly energy-conserving for B-only: std(KE) / mean(KE) must be < 1e-4.
    """
    print("Test 4: Uniform B only  (energy conservation, E=0 regression)")
    VALDATA.mkdir(parents=True, exist_ok=True)
    nx = ny = nz = 16
    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)
    B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    B[..., 2] = 1.0    # Bz = 1 T
    E = np.zeros_like(B)
    write_bfld(VALDATA / "t4_B.bfld", B, E, bounds)
    write_config(VALDATA / "t4_B.json", "t4_B.bfld",
                 n_particles=50_000, detector_center_mm=(110.0, 0.0, 0.0))
    out = VALOUT / "t4_B"

    if not run_batch(VALDATA / "t4_B.json", out):
        print("   tracer error")
        return False

    hits = read_hits(out)
    if not hits:
        print("   no hits recorded")
        return False

    # ke_MeV column is already in MeV.
    # Use numpy for numerically stable mean/std — Python's sum() accumulates
    # O(N * eps_f64) error over 50k identical values, producing a spurious
    # nonzero std even when all values are bit-identical (true std = 0).
    ke_arr  = np.array([h[2] for h in hits])
    mean_ke = float(np.mean(ke_arr))
    std_ke  = float(np.std(ke_arr))
    rel_std = std_ke / mean_ke if mean_ke > 0 else float("inf")
    n_unique = int(len(np.unique(ke_arr)))
    tol     = 1e-4

    ok = rel_std <= tol
    print(f"   hits = {len(hits)},  kinetic energy mean = {mean_ke:.4f} MeV,  "
          f"std/mean = {rel_std:.2e},  unique values = {n_unique}")
    if not ok:
        print(f"   FAIL: energy not conserved (std/mean = {rel_std:.2e} > {tol:.0e})")

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test4_uniform_B"] = {
        "pass": ok,
        "kinetic_energy_mean_MeV": round(mean_ke, 6),
        "energy_rel_std": float(f"{rel_std:.3e}"),
        "n_unique_ke_values": n_unique,
        "tolerance": tol,
        "note": (
            "Uses relativistic Boris integrator. Particles store u = γv; "
            "KE at detector = (γ-1)m_p c². In a uniform B field all particles "
            "execute the same GPU ops → identical float32 KE (n_unique = 1). "
            "True std = 0; residual rel_std ≈ f64 machine precision."
        ),
        **png_info,
    }
    return ok


# ── test 5: pencil source, tilted 2°, zero field ─────────────────────────────

def test5_pencil_tilted():
    """
    Pencil source at (-0.10, 0, 0), direction tilted 2° toward +y.
    B = E = 0.  Protons travel in a straight line, so:
        mean_y ≈ 210 mm * tan(2°) ≈ 7.333 mm
        std(y)  < 0.1 mm  (all particles are identical)
    """
    print("Test 5: Pencil source, 2° tilt  (straight-line, delta-beam)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    import math
    tilt_rad = math.radians(2.0)
    # Direction: mostly +x, small +y component
    dx = math.cos(tilt_rad)
    dy = math.sin(tilt_rad)

    nx = ny = nz = 16
    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)
    B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    E = np.zeros_like(B)
    write_bfld(VALDATA / "t5_pencil.bfld", B, E, bounds)

    cfg = {
        "field_path": "t5_pencil.bfld",
        "detector": {
            "center_mm": [110.0, 0.0, 0.0],
            "normal":    [1.0, 0.0, 0.0],
            "up":        [0.0, 1.0, 0.0],
            "width_mm":  500.0,
            "height_mm": 500.0,
            "pixels":    [512, 512],
        },
        "source": {
            "source_type": "pencil",
            "n_particles": 10_000,
            "energy_MeV":  14.7,
            "position_mm": [-100.0, 0.0, 0.0],
            "direction":   [dx, dy, 0.0],
        },
        "dt_ps":     1.0,
        "max_steps": 20_000,
    }
    import json
    cfg_path = VALDATA / "t5_pencil.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    out = VALOUT / "t5_pencil"
    if not run_batch(cfg_path, out):
        print("   tracer error")
        return False

    hits = read_hits(out)
    if not hits:
        print("   no hits recorded")
        return False

    ys = [h[0] for h in hits]
    mean_y = _mean(ys)
    std_y  = _std(ys, mean_y)

    # Analytic: source at x=-0.10, detector at x=0.11 → total Δx = 0.21 m
    expected_y_mm = 210.0 * math.tan(tilt_rad)   # ≈ 7.333 mm

    rel_err = abs(mean_y - expected_y_mm) / expected_y_mm
    ok_mean = rel_err < 0.01          # within 1 %
    ok_std  = std_y < 0.1             # delta-beam: all particles identical

    ok = ok_mean and ok_std
    print(f"   hits = {len(hits)},  mean_y = {mean_y:+.4f} mm  "
          f"(expected {expected_y_mm:+.4f} mm,  rel_err = {rel_err:.4f}),  "
          f"std_y = {std_y:.4f} mm")
    if not ok_mean:
        print(f"   FAIL: mean_y rel_err {rel_err:.4f} > 1 %")
    if not ok_std:
        print(f"   FAIL: std_y {std_y:.4f} mm ≥ 0.1 mm (should be zero for pencil)")

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test5_pencil_tilted"] = {
        "pass": ok,
        "mean_y_mm": round(mean_y, 4),
        "expected_y_mm": round(expected_y_mm, 4),
        "relative_error": round(rel_err, 6),
        "std_y_mm": round(std_y, 6),
        **png_info,
    }
    return ok


# ── test 6: point source, cone covers full detector ──────────────────────────

def test6_point_full_cone():
    """
    Point source at (-100, 0, 0) mm, direction +x, cone_half_angle = 45°.
    B = E = 0.  Detector at x=110 mm, 500×500 mm.

    Geometry: max lateral offset at detector plane = 210 mm * tan(45°) = 210 mm.
    The detector extends ±250 mm in y and z, so every ray in the cone
    satisfies |y| ≤ 210 < 250 and |z| ≤ 210 < 250 → ALL particles hit, so
        hit_fraction = hits / n_particles ≥ 0.99.
    """
    print("Test 6: Point source, 45° cone, all particles hit detector  (hit fraction ≈ 1)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    n_particles = 50_000
    nx = ny = nz = 16
    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)
    B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    E = np.zeros_like(B)
    write_bfld(VALDATA / "t6_point.bfld", B, E, bounds)

    cfg = {
        "field_path": "t6_point.bfld",
        "detector": {
            "center_mm": [110.0, 0.0, 0.0],
            "normal":    [1.0, 0.0, 0.0],
            "up":        [0.0, 1.0, 0.0],
            "width_mm":  500.0,
            "height_mm": 500.0,
            "pixels":    [512, 512],
        },
        "source": {
            "source_type":        "point",
            "n_particles":        n_particles,
            "energy_MeV":         14.7,
            "position_mm":        [-100.0, 0.0, 0.0],
            "direction":          [1.0, 0.0, 0.0],
            "cone_half_angle_deg": 45.0,
        },
        "dt_ps":     1.0,
        "max_steps": 20_000,
    }
    import json
    cfg_path = VALDATA / "t6_point.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    out = VALOUT / "t6_point"
    if not run_batch(cfg_path, out):
        print("   tracer error")
        return False

    hits = read_hits(out)
    n_hits = len(hits)
    hit_fraction = n_hits / n_particles

    ok = hit_fraction >= 0.99
    print(f"   hits = {n_hits} / {n_particles},  fraction = {hit_fraction:.4f}")
    if not ok:
        print(f"   FAIL: hit fraction {hit_fraction:.4f} < 0.99  "
              "(cone should fully cover the 500×500 mm detector)")

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test6_point_full_cone"] = {
        "pass": ok,
        "hits": n_hits,
        "n_particles": n_particles,
        "hit_fraction": round(hit_fraction, 6),
        "threshold": 0.99,
        **png_info,
    }
    return ok


# ── test 7: disk source, zero spread, spatial std matches disk radius ─────────

def test7_disk_spatial_spread():
    """
    Disk source at (-100, 0, 0) mm, radius = 30 mm, direction +x, cone = 0°.
    B = E = 0.  Detector at x=110 mm.

    With zero cone angle particles travel in straight lines, so the disk
    projects directly onto the detector with the same radial distribution.
    For a uniform disk of radius R: std of one Cartesian component = R/2.
    With R = 30 mm: expected std = 15.00 mm.

    Checks:
      1. mean_y and mean_z near 0 (< 1 mm)
      2. std_y and std_z within 5 % of 30 / sqrt(2)
    """
    print("Test 7: Disk source, 0° cone  (disk projects to detector, std = R/2)")
    import math
    VALDATA.mkdir(parents=True, exist_ok=True)

    radius_mm = 30.0
    nx = ny = nz = 16
    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)
    B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    E = np.zeros_like(B)
    write_bfld(VALDATA / "t7_disk.bfld", B, E, bounds)

    cfg = {
        "field_path": "t7_disk.bfld",
        "detector": {
            "center_mm": [110.0, 0.0, 0.0],
            "normal":    [1.0, 0.0, 0.0],
            "up":        [0.0, 1.0, 0.0],
            "width_mm":  500.0,
            "height_mm": 500.0,
            "pixels":    [512, 512],
        },
        "source": {
            "source_type":        "disk",
            "n_particles":        100_000,
            "energy_MeV":         14.7,
            "center_mm":          [-100.0, 0.0, 0.0],
            "direction":          [1.0, 0.0, 0.0],
            "radius_um":          radius_mm * 1000.0,   # 30 mm → 30 000 µm
            "cone_half_angle_deg": 0.0,
        },
        "dt_ps":     1.0,
        "max_steps": 20_000,
    }
    import json
    cfg_path = VALDATA / "t7_disk.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    out = VALOUT / "t7_disk"
    if not run_batch(cfg_path, out):
        print("   tracer error")
        return False

    hits = read_hits(out)
    if not hits:
        print("   no hits recorded")
        return False

    ys = [h[0] for h in hits]
    zs = [h[1] for h in hits]
    mean_y, mean_z = _mean(ys), _mean(zs)
    std_y,  std_z  = _std(ys, mean_y), _std(zs, mean_z)

    expected_std = radius_mm / 2.0   # std of one Cartesian component over a uniform disk = R/2

    ok_mean = abs(mean_y) < 1.0 and abs(mean_z) < 1.0
    ok_std  = (abs(std_y - expected_std) / expected_std < 0.05 and
               abs(std_z - expected_std) / expected_std < 0.05)
    ok = ok_mean and ok_std

    print(f"   hits = {len(hits)},  "
          f"mean_y = {mean_y:+.3f} mm,  mean_z = {mean_z:+.3f} mm")
    print(f"   std_y = {std_y:.3f} mm,  std_z = {std_z:.3f} mm  "
          f"(expected {expected_std:.3f} mm)")

    if not ok_mean:
        print(f"   FAIL: mean |y| or |z| ≥ 1 mm (should be ≈ 0)")
    if not ok_std:
        print(f"   FAIL: std outside 5 % of {expected_std:.3f} mm")

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test7_disk_spatial_spread"] = {
        "pass": ok,
        "mean_y_mm": round(mean_y, 4),
        "mean_z_mm": round(mean_z, 4),
        "std_y_mm":  round(std_y, 4),
        "std_z_mm":  round(std_z, 4),
        "expected_std_mm": round(expected_std, 4),
        **png_info,
    }
    return ok


# ── test 8: energy spread ────────────────────────────────────────────────────

def test8_energy_spread():
    """Gaussian energy spread: std(ke_MeV)/mean(ke_MeV) ≈ spread_percent/100."""
    print("Test 8: Energy spread  (pencil, 5% Gaussian)")
    out = VALOUT / "t8_energy_spread"

    # Build a minimal field (no B, no E) so protons pass straight through
    bfld = VALDATA / "t8_zero.bfld"
    VALDATA.mkdir(parents=True, exist_ok=True)
    B = np.zeros((2, 2, 2, 3), dtype=np.float32)
    E = np.zeros((2, 2, 2, 3), dtype=np.float32)
    bounds = (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05)
    write_bfld(bfld, B, E, bounds)

    SPREAD = 5.0  # percent
    SEED   = 42

    cfg_path = VALDATA / "t8_energy_spread.json"
    cfg = {
        "field_path": str(bfld.resolve()),
        "detector": {
            "center_mm": [110.0, 0.0, 0.0],
            "normal":    [1.0, 0.0, 0.0],
            "up":        [0.0, 1.0, 0.0],
            "width_mm":  500.0,
            "height_mm": 500.0,
            "pixels":    [512, 512],
        },
        "source": {
            "source_type":           "pencil",
            "n_particles":           10000,
            "energy_MeV":            14.7,
            "energy_spread_percent": SPREAD,
            "seed":                  SEED,
            "position_mm":           [-100.0, 0.0, 0.0],
            "direction":             [1.0, 0.0, 0.0],
        },
        "dt_ps":     2.0,
        "max_steps": 20000,
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    if not run_batch(cfg_path, out):
        REPORT["test8_energy_spread"] = {"pass": False, "error": "simulation failed"}
        return False

    hits = read_hits(out)
    ok = True

    # Energy spread check: std(ke_MeV) / mean(ke_MeV) * 100 ≈ SPREAD ± 1%
    ke_vals = [h[2] for h in hits]
    if not ke_vals:
        print("   No hits — cannot check energy spread")
        ok = False
        spread_measured = None
    else:
        mean_ke = sum(ke_vals) / len(ke_vals)
        std_ke  = (sum((k - mean_ke)**2 for k in ke_vals) / len(ke_vals))**0.5
        spread_measured = std_ke / mean_ke * 100.0
        tol = 1.0  # percentage points
        spread_ok = abs(spread_measured - SPREAD) <= tol
        if not spread_ok:
            print(f"   Energy spread {spread_measured:.2f}% ≠ {SPREAD}% ± {tol}%")
        ok = ok and spread_ok
        print(f"   ke mean={mean_ke:.4f} MeV  std={std_ke:.4f} MeV  "
              f"spread={spread_measured:.2f}% (target {SPREAD}%)")

    # Also verify all KE values differ (not monoenergetic)
    if ke_vals and len(set(round(k, 4) for k in ke_vals)) < 2:
        print("   All ke_MeV values identical — energy spread not applied")
        ok = False

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok

    REPORT["test8_energy_spread"] = {
        "pass":              ok,
        "n_hits":            len(hits),
        "spread_target_pct": SPREAD,
        "spread_meas_pct":   round(spread_measured, 3) if spread_measured else None,
        **png_info,
    }
    return ok


# ── test 9: Gaussian blur — count conservation + spot widening ───────────────

def test9_blur_conservation():
    """
    Pencil beam (no field, no energy spread) with large Gaussian PSF blur and
    no Poisson noise.

    Checks:
      1. Total count is conserved: sum(processed) ≈ sum(raw) within 2%.
         Gaussian blur with clamp-to-edge is a linear normalised filter.
      2. Spot grows: σ(processed) > σ(raw) in both y and z,
         and the measured σ is within a factor of 2 of the expected pixel sigma.
    """
    print("Test 9: Gaussian blur  (count conservation + spot widening)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    BLUR_SIGMA_UM = 3000.0   # 3 mm
    N = 10_000

    bfld = VALDATA / "t9_zero.bfld"
    B = np.zeros((2, 2, 2, 3), dtype=np.float32)
    E = np.zeros_like(B)
    write_bfld(bfld, B, E, (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05))

    cfg = {
        "field_path": str(bfld.resolve()),
        "detector": {
            "center_mm": [110.0, 0.0, 0.0],
            "normal":    [1.0, 0.0, 0.0],
            "up":        [0.0, 1.0, 0.0],
            "width_mm":  500.0,
            "height_mm": 500.0,
            "pixels":    [512, 512],
        },
        "source": {
            "source_type": "pencil",
            "n_particles": N,
            "energy_MeV":  14.7,
            "position_mm": [-100.0, 0.0, 0.0],
            "direction":   [1.0, 0.0, 0.0],
        },
        "detector_response": {
            "blur_sigma_um":    BLUR_SIGMA_UM,
            "background_counts": 0.0,
            "poisson_noise":    False,
        },
        "dt_ps":     2.0,
        "max_steps": 20_000,
    }
    cfg_path = VALDATA / "t9_blur.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    out = VALOUT / "t9_blur"
    if not run_batch(cfg_path, out):
        REPORT["test9_blur_conservation"] = {"pass": False, "error": "simulation failed"}
        return False

    raw_bins  = sorted(out.glob("radiograph_*_raw_counts.bin"))
    proc_bins = sorted(out.glob("radiograph_*_processed_counts.bin"))
    if not raw_bins or not proc_bins:
        print("   missing count .bin files")
        REPORT["test9_blur_conservation"] = {"pass": False, "error": "missing bin files"}
        return False

    raw  = np.frombuffer(raw_bins[0].read_bytes(),  dtype="<u4").reshape(1024, 1024).astype(np.float64)
    proc = np.frombuffer(proc_bins[0].read_bytes(), dtype="<f4").reshape(1024, 1024).astype(np.float64)

    raw_sum  = float(raw.sum())
    proc_sum = float(proc.sum())
    frac_diff = abs(proc_sum - raw_sum) / max(raw_sum, 1.0)
    count_ok = frac_diff < 0.02

    # Weighted 1D sigma along each axis via marginal projections.
    # col axis (index 1) ↔ y_mm;  row axis (index 0) ↔ z_mm.
    cols = np.arange(1024, dtype=np.float64)
    rows = np.arange(1024, dtype=np.float64)

    def sigma_1d(grid, indices, projection_axis):
        proj  = grid.sum(axis=projection_axis)
        total = proj.sum()
        if total < 1.0:
            return 0.0
        mean = (proj * indices).sum() / total
        return float(np.sqrt(((proj * (indices - mean) ** 2).sum()) / total))

    raw_sig_y  = sigma_1d(raw,  cols, projection_axis=0)
    raw_sig_z  = sigma_1d(raw,  rows, projection_axis=1)
    proc_sig_y = sigma_1d(proc, cols, projection_axis=0)
    proc_sig_z = sigma_1d(proc, rows, projection_axis=1)

    # GPU texture: 1024 px over 500 mm → pitch ≈ 488 µm/px
    pitch_um = 500.0 / 1024.0 * 1000.0
    expected_sig_px = BLUR_SIGMA_UM / pitch_um   # ≈ 6.1 px

    width_ok = (
        proc_sig_y > raw_sig_y
        and proc_sig_z > raw_sig_z
        and proc_sig_y > 0.5 * expected_sig_px
        and proc_sig_z > 0.5 * expected_sig_px
    )

    ok = count_ok and width_ok
    print(f"   raw_sum={raw_sum:.0f}  proc_sum={proc_sum:.1f}  frac_diff={frac_diff:.4f}")
    print(f"   raw σ_y={raw_sig_y:.2f}px  proc σ_y={proc_sig_y:.2f}px  (expected ≥{0.5*expected_sig_px:.2f}px)")
    print(f"   raw σ_z={raw_sig_z:.2f}px  proc σ_z={proc_sig_z:.2f}px")
    if not count_ok:
        print(f"   FAIL: count not conserved (frac_diff={frac_diff:.4f} > 0.02)")
    if not width_ok:
        print(f"   FAIL: spot did not widen as expected after blur")

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test9_blur_conservation"] = {
        "pass":              ok,
        "raw_sum":           raw_sum,
        "proc_sum":          round(proc_sum, 2),
        "frac_diff":         round(frac_diff, 6),
        "raw_sig_y_px":      round(raw_sig_y, 3),
        "raw_sig_z_px":      round(raw_sig_z, 3),
        "proc_sig_y_px":     round(proc_sig_y, 3),
        "proc_sig_z_px":     round(proc_sig_z, 3),
        "expected_sig_px":   round(expected_sig_px, 3),
        "blur_sigma_um":     BLUR_SIGMA_UM,
        **png_info,
    }
    return ok


# ── test 10: Poisson reproducibility ─────────────────────────────────────────

def test10_poisson_reproducibility():
    """
    Same raw counts + same noise_seed → byte-identical processed_counts.bin.
    Different seed → different processed_counts.bin.

    Uses a pencil beam on a zero field so raw counts are fully deterministic
    (all N particles land in the same pixel, regardless of spatial sampling).
    """
    print("Test 10: Poisson reproducibility  (same seed → identical, diff seed → different)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    N    = 5_000
    BLUR = 500.0   # 0.5 mm, ~1 px at 488 µm/px — small blur to keep a bright spot

    bfld = VALDATA / "t9_zero.bfld"
    if not bfld.exists():
        B = np.zeros((2, 2, 2, 3), dtype=np.float32)
        E = np.zeros_like(B)
        write_bfld(bfld, B, E, (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05))

    def _run(seed, tag):
        cfg = {
            "field_path": str(bfld.resolve()),
            "detector": {
                "center_mm": [110.0, 0.0, 0.0],
                "normal":    [1.0, 0.0, 0.0],
                "up":        [0.0, 1.0, 0.0],
                "width_mm":  500.0,
                "height_mm": 500.0,
                "pixels":    [512, 512],
            },
            "source": {
                "source_type": "pencil",
                "n_particles": N,
                "energy_MeV":  14.7,
                "position_mm": [-100.0, 0.0, 0.0],
                "direction":   [1.0, 0.0, 0.0],
            },
            "detector_response": {
                "blur_sigma_um":    BLUR,
                "background_counts": 0.0,
                "poisson_noise":    True,
                "noise_seed":       seed,
            },
            "dt_ps":     2.0,
            "max_steps": 20_000,
        }
        cfg_path = VALDATA / f"t10_{tag}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        out = VALOUT / f"t10_{tag}"
        if not run_batch(cfg_path, out):
            return None
        bins = sorted(out.glob("radiograph_*_processed_counts.bin"))
        return bins[0].read_bytes() if bins else None

    data_a = _run(42, "seed42_a")
    data_b = _run(42, "seed42_b")
    data_c = _run(99, "seed99")

    if data_a is None or data_b is None or data_c is None:
        print("   simulation failed for one or more runs")
        REPORT["test10_poisson_reproducibility"] = {"pass": False, "error": "simulation failed"}
        return False

    same_ok = (data_a == data_b)
    diff_ok = (data_a != data_c)
    ok = same_ok and diff_ok

    print(f"   seed=42 run A vs run B identical: {same_ok}")
    print(f"   seed=42 vs seed=99 different:     {diff_ok}")
    if not same_ok:
        print("   FAIL: same seed produced different output — Poisson RNG not deterministic")
    if not diff_ok:
        print("   FAIL: different seeds produced identical output — seed has no effect")

    REPORT["test10_poisson_reproducibility"] = {
        "pass":           ok,
        "same_seed_identical": same_ok,
        "diff_seed_differs":   diff_ok,
    }
    return ok


# ── test 11: exponential / TNSA energy spectrum ──────────────────────────────

def test11_exponential_spectrum():
    """
    Pencil source, B = E = 0, exponential spectrum T = 3 MeV, cutoff = 40 MeV.

    For dN/dE ∝ exp(−E/T) with cutoff ≫ T the mean is close to T.
    Checks:
      1. All ke_MeV ≤ cutoff + 0.05 MeV (hard cutoff enforced)
      2. mean(ke_MeV) within 35 % of T (correct distribution shape)
      3. std(ke_MeV) / mean(ke_MeV) > 0.3  (not monoenergetic)
    """
    print("Test 11: Exponential / TNSA spectrum  (T=3 MeV, cutoff=40 MeV)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    T_MEV     = 3.0
    CUTOFF    = 40.0
    N         = 20_000

    bfld = VALDATA / "t11_zero.bfld"
    if not bfld.exists():
        B = np.zeros((2, 2, 2, 3), dtype=np.float32)
        E = np.zeros_like(B)
        write_bfld(bfld, B, E, (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05))

    cfg = {
        "field_path": str(bfld.resolve()),
        "detector": {
            "center_mm": [110.0, 0.0, 0.0],
            "normal":    [1.0, 0.0, 0.0],
            "up":        [0.0, 1.0, 0.0],
            "width_mm":  500.0,
            "height_mm": 500.0,
            "pixels":    [512, 512],
        },
        "source": {
            "source_type":    "pencil",
            "n_particles":    N,
            "energy_MeV":     14.7,  # nominal; overridden by spectrum
            "temperature_MeV": T_MEV,
            "cutoff_MeV":     CUTOFF,
            "position_mm":    [-100.0, 0.0, 0.0],
            "direction":      [1.0, 0.0, 0.0],
        },
        "dt_ps":     1.0,
        "max_steps": 20_000,
    }
    cfg_path = VALDATA / "t11_tnsa.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    out = VALOUT / "t11_tnsa"
    if not run_batch(cfg_path, out):
        REPORT["test11_exponential_spectrum"] = {"pass": False, "error": "simulation failed"}
        return False

    hits = read_hits(out)
    if not hits:
        print("   no hits recorded")
        REPORT["test11_exponential_spectrum"] = {"pass": False, "error": "no hits"}
        return False

    ke_vals = [h[2] for h in hits]
    mean_ke = _mean(ke_vals)
    std_ke  = _std(ke_vals, mean_ke)
    max_ke  = max(ke_vals)

    cutoff_ok = max_ke <= CUTOFF + 0.05
    mean_ok   = abs(mean_ke - T_MEV) / T_MEV < 0.35   # within 35 % of T
    spread_ok = (std_ke / mean_ke) > 0.3               # not monoenergetic

    ok = cutoff_ok and mean_ok and spread_ok
    print(f"   hits={len(ke_vals)}  mean={mean_ke:.4f} MeV  std={std_ke:.4f} MeV  max={max_ke:.4f} MeV")
    print(f"   (T={T_MEV} MeV, cutoff={CUTOFF} MeV)")
    if not cutoff_ok:
        print(f"   FAIL: max_ke {max_ke:.4f} > cutoff {CUTOFF} + 0.05 MeV")
    if not mean_ok:
        print(f"   FAIL: mean {mean_ke:.4f} not within 35% of T={T_MEV}")
    if not spread_ok:
        print(f"   FAIL: std/mean {std_ke/mean_ke:.3f} < 0.3 (should be spread, not mono)")

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test11_exponential_spectrum"] = {
        "pass":           ok,
        "n_hits":         len(ke_vals),
        "mean_ke_MeV":    round(mean_ke, 4),
        "std_ke_MeV":     round(std_ke, 4),
        "max_ke_MeV":     round(max_ke, 4),
        "temperature_MeV": T_MEV,
        "cutoff_MeV":     CUTOFF,
        **png_info,
    }
    return ok


# ── test 12: relativistic energy conservation at 60 MeV ──────────────────────

def test12_relativistic_60mev():
    """
    Pencil source, 60 MeV, zero B and E field.

    At 60 MeV γ ≈ 1.064 (6.4% relativistic correction). Wrong kinetic energy
    initialisation (e.g. using classical KE = ½mv²) would give γ ≈ 1.032 and
    an impact KE ≈ 58.17 MeV — a detectable ~1.8 MeV shift.

    Checks:
      1. mean(KE) = 60.000 ± 0.1 MeV  (within 0.17%)
      2. std / mean < 1e-4             (monoenergetic — no spread introduced)
    """
    print("Test 12: Relativistic 60 MeV energy conservation  (γ ≈ 1.064)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    E_MEV = 60.0
    N     = 10_000

    bfld = VALDATA / "t12_zero.bfld"
    if not bfld.exists():
        B = np.zeros((2, 2, 2, 3), dtype=np.float32)
        E = np.zeros_like(B)
        write_bfld(bfld, B, E, (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05))

    cfg = {
        "field_path": str(bfld.resolve()),
        "detector": {
            "center_mm": [110.0, 0.0, 0.0],
            "normal":    [1.0, 0.0, 0.0],
            "up":        [0.0, 1.0, 0.0],
            "width_mm":  500.0,
            "height_mm": 500.0,
            "pixels":    [512, 512],
        },
        "source": {
            "source_type": "pencil",
            "n_particles": N,
            "energy_MeV":  E_MEV,
            "position_mm": [-100.0, 0.0, 0.0],
            "direction":   [1.0, 0.0, 0.0],
        },
        "dt_ps":     1.0,
        "max_steps": 5_000,
    }
    cfg_path = VALDATA / "t12_relat.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    out = VALOUT / "t12_relat"
    if not run_batch(cfg_path, out):
        REPORT["test12_relativistic_60mev"] = {"pass": False, "error": "simulation failed"}
        return False

    hits = read_hits(out)
    if not hits:
        print("   no hits recorded")
        REPORT["test12_relativistic_60mev"] = {"pass": False, "error": "no hits"}
        return False

    ke_arr   = np.array([h[2] for h in hits])
    mean_ke  = float(np.mean(ke_arr))
    std_ke   = float(np.std(ke_arr))
    rel_std  = std_ke / mean_ke
    n_unique = int(len(np.unique(ke_arr)))

    mean_ok   = abs(mean_ke - E_MEV) < 0.1     # within 0.1 MeV of 60
    spread_ok = rel_std < 1e-4                  # monoenergetic

    ok = mean_ok and spread_ok
    print(f"   hits={len(ke_arr)}  mean={mean_ke:.4f} MeV  std={std_ke:.4f} MeV  rel_std={rel_std:.2e}")
    if not mean_ok:
        print(f"   FAIL: mean {mean_ke:.4f} MeV not within 0.1 MeV of {E_MEV} (non-relativistic init would give ~58.17 MeV)")
    if not spread_ok:
        print(f"   FAIL: rel_std {rel_std:.2e} >= 1e-4 (monoenergetic source should not spread)")

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test12_relativistic_60mev"] = {
        "pass":             ok,
        "n_hits":           len(ke_arr),
        "mean_ke_MeV":      round(mean_ke, 4),
        "std_ke_MeV":       round(std_ke, 6),
        "rel_std":          round(rel_std, 8),
        "n_unique_ke_values": n_unique,
        "target_MeV":       E_MEV,
        **png_info,
    }
    return ok


# ── test 13: tilted geometry — beam along +z, detector facing -z ─────────────

def test13_tilted_geometry():
    """
    Parallel beam along +z (not the default +x), zero field.
    Source at (0, 0, -100mm), detector at (0, 0, +110mm) facing -z.
    Beam radius = 150mm, well inside the 500mm detector (half-extent 250mm),
    so with correct domain-exit logic every particle should reach the detector.

    The field extends ±60mm in x/y, so the old axis-biased margin (margin.x = 60mm)
    made domain_max.x = 120mm.  Particles at |x| > 120mm were killed immediately on
    step 1 — that is ~10% of a 150mm-radius beam.

    Checks:
      1. hit_fraction >= 0.99  (no particles killed by domain exit)
      2. std(local-y) ≈ std(local-z) ≈ 75mm  (uniform disk R=150mm, std = R/2)
    """
    print("Test 13: Tilted geometry  (+z beam, detector facing -z, 150mm radius)")
    import math, json as _json
    VALDATA.mkdir(parents=True, exist_ok=True)

    n_particles = 50_000
    radius_mm   = 150.0

    nx = ny = nz = 16
    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)
    B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    E = np.zeros_like(B)
    write_bfld(VALDATA / "t13_zero.bfld", B, E, bounds)

    cfg = {
        "field_path": "t13_zero.bfld",
        "detector": {
            "center_mm": [0.0, 0.0, 110.0],
            "normal":    [0.0, 0.0, -1.0],   # facing -z (toward source)
            "up":        [0.0, 1.0,  0.0],
            "width_mm":  500.0,
            "height_mm": 500.0,
            "pixels":    [512, 512],
        },
        "source": {
            "source_type":    "parallel",
            "n_particles":    n_particles,
            "energy_MeV":     14.7,
            "beam_center":    [0.0, 0.0, -0.1],   # 100mm upstream in z [m]
            "beam_direction": [0.0, 0.0,  1.0],   # +z beam axis
            "beam_radius_mm": radius_mm,
            "angular_spread_deg": 0.0,
        },
        "dt_ps":     1.0,
        "max_steps": 20_000,
    }
    cfg_path = VALDATA / "t13_tilted.json"
    with open(cfg_path, "w") as f:
        _json.dump(cfg, f, indent=2)

    out = VALOUT / "t13_tilted"
    if not run_batch(cfg_path, out):
        print("   tracer error")
        REPORT["test13_tilted_geometry"] = {"pass": False, "error": "simulation failed"}
        return False

    hits = read_hits(out)
    n_hits       = len(hits)
    hit_fraction = n_hits / n_particles

    # Detector basis for +z beam, normal=[0,0,-1], up=[0,1,0]:
    #   u_y = [0,1,0]  (world-y ↔ local-y)
    #   v_z = cross([0,0,-1],[0,1,0]) = [1,0,0]  (world-x ↔ local-z)
    # So h[0] = world-y offset,  h[1] = world-x offset.
    # Uniform disk R=150mm → std of one Cartesian component = R/2 = 75mm.
    expected_std = radius_mm / 2.0

    ys = [h[0] for h in hits]
    zs = [h[1] for h in hits]
    std_y = _std(ys) if ys else 0.0
    std_z = _std(zs) if zs else 0.0

    hit_ok  = hit_fraction >= 0.99
    std_ok  = (abs(std_y - expected_std) / expected_std < 0.05 and
               abs(std_z - expected_std) / expected_std < 0.05)
    ok = hit_ok and std_ok

    print(f"   hits = {n_hits} / {n_particles},  fraction = {hit_fraction:.4f}")
    print(f"   std_y = {std_y:.2f} mm,  std_z = {std_z:.2f} mm  (expected {expected_std:.1f} mm)")
    if not hit_ok:
        print(f"   FAIL: hit fraction {hit_fraction:.4f} < 0.99  "
              "(domain-exit bug kills outer-ring particles without the shader fix)")
    if not std_ok:
        print(f"   FAIL: spatial std outside 5% of {expected_std:.1f} mm")

    png_ok, png_info = check_png_output(out)
    ok = ok and png_ok
    REPORT["test13_tilted_geometry"] = {
        "pass":           ok,
        "hits":           n_hits,
        "n_particles":    n_particles,
        "hit_fraction":   round(hit_fraction, 6),
        "std_y_mm":       round(std_y, 2),
        "std_z_mm":       round(std_z, 2),
        "expected_std_mm": expected_std,
        **png_info,
    }
    return ok


# ── test 14: superimposed fields ─────────────────────────────────────────────

def test14_superimposed_fields():
    """
    Primary field: B = E = 0 (2×2×2 grid, same spatial bounds as test4).
    Extra field:   Bz = 1 T (16×16×16 grid, same bounds).

    After CPU compositing the effective field is Bz = 1 T — identical to test4.
    Energy conservation check: std(KE) / mean(KE) < 1e-4.

    This validates:
      - extra field is loaded and resampled onto the primary grid
      - result is physics-equivalent to having a single Bz=1T field
    """
    print("Test 14: Superimposed fields  (zero primary + Bz=1T extra = uniform Bz=1T)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)

    # Primary: all zeros, coarse grid
    B_zero = np.zeros((2, 2, 2, 3), dtype=np.float32)
    E_zero = np.zeros_like(B_zero)
    write_bfld(VALDATA / "t14_zero.bfld", B_zero, E_zero, bounds)

    # Extra: Bz = 1 T, fine grid
    B_bz = np.zeros((16, 16, 16, 3), dtype=np.float32)
    B_bz[..., 2] = 1.0
    E_bz = np.zeros_like(B_bz)
    write_bfld(VALDATA / "t14_Bz.bfld", B_bz, E_bz, bounds)

    deck = f"""\
[field]
path = "t14_zero.bfld"
scale_B = 1.0
scale_E = 0.0

[[field.extra_b]]
path = "t14_Bz.bfld"
scale_B = 1.0
scale_E = 0.0

[source]
type = "parallel"
n_particles = 50000
energy_MeV = 14.7
beam_radius_mm = 30.0
source_distance_mm = 100.0

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [512, 512]

[numerics]
dt_ps = 1.0
max_steps = 20000
"""
    deck_path = VALDATA / "t14_superimpose.toml"
    deck_path.write_text(deck)

    out = VALOUT / "t14_superimpose"
    import shutil
    if out.exists():
        shutil.rmtree(out)

    env = os.environ.copy()
    brew_lib = Path("/opt/homebrew/lib")
    icd      = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    if icd.exists() and "VK_ICD_FILENAMES" not in env:
        env["VK_ICD_FILENAMES"] = str(icd)
    if brew_lib.exists():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        if str(brew_lib) not in existing:
            env["DYLD_LIBRARY_PATH"] = (str(brew_lib) + ":" + existing).rstrip(":")

    cmd = [str(BIN), "run", str(deck_path), "-o", str(out)]
    try:
        result = subprocess.run(cmd, cwd=VALDATA, capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        REPORT["test14_superimposed_fields"] = {"pass": False, "error": "timeout"}
        return False
    if result.returncode != 0:
        print(f"   tracer error: {result.stderr[-300:]}")
        REPORT["test14_superimposed_fields"] = {"pass": False, "error": "simulation failed"}
        return False

    hits_bin = out / "counts" / "hits.bin"
    if not hits_bin.exists():
        print("   no hits.bin found")
        REPORT["test14_superimposed_fields"] = {"pass": False, "error": "no hits.bin"}
        return False
    raw = hits_bin.read_bytes()
    if len(raw) < 4:
        print("   hits.bin too short")
        REPORT["test14_superimposed_fields"] = {"pass": False, "error": "no hits"}
        return False
    n_hits_recorded = struct.unpack_from("<I", raw, 0)[0]
    hits_data = np.frombuffer(raw, dtype="<f4", offset=4).reshape(-1, 3)
    ke_arr = hits_data[:, 2]
    mean_ke = float(np.mean(ke_arr))
    std_ke  = float(np.std(ke_arr))
    rel_std = std_ke / mean_ke if mean_ke > 0 else float("inf")
    tol = 1e-4

    ok = rel_std <= tol
    print(f"   hits = {len(ke_arr)},  mean KE = {mean_ke:.4f} MeV,  std/mean = {rel_std:.2e}")
    if not ok:
        print(f"   FAIL: std/mean = {rel_std:.2e} > {tol:.0e}  (B-only → energy must be conserved)")

    REPORT["test14_superimposed_fields"] = {
        "pass": ok,
        "n_hits": int(len(ke_arr)),
        "mean_ke_MeV": round(mean_ke, 6),
        "std_mean_ratio": float(f"{rel_std:.3e}"),
        "tolerance": tol,
    }
    return ok


# ── test 15: adaptive dt gives same physics as fixed dt ──────────────────────

def test15_adaptive_dt():
    """
    Pencil source through uniform Bz = 1 T.  Run twice:
      A) fixed dt_ps = 1.0  (explicit, disables adaptive schedule)
      B) no dt_ps supplied  (triggers adaptive schedule)

    Checks:
      1. Both runs produce ≥ 10000 hits
      2. Mean hit positions agree within 1 mm  (adaptive dt gives same deflection)
      3. Energy conservation holds in both  (std/mean < 1e-4)
    """
    print("Test 15: Adaptive dt — same physics as fixed dt")
    VALDATA.mkdir(parents=True, exist_ok=True)

    N_PARTICLES = 20_000
    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)

    # Bz = 1 T (same field as test 4)
    bfld = VALDATA / "t15_Bz.bfld"
    if not bfld.exists():
        B = np.zeros((16, 16, 16, 3), dtype=np.float32)
        B[..., 2] = 1.0
        E = np.zeros_like(B)
        write_bfld(bfld, B, E, bounds)

    def make_deck(with_fixed_dt):
        dt_line = "dt_ps = 1.0\n" if with_fixed_dt else ""
        return f"""\
[field]
path = "t15_Bz.bfld"
scale_B = 1.0
scale_E = 0.0

[source]
type = "pencil"
n_particles = {N_PARTICLES}
energy_MeV = 14.7
position_mm = [-100.0, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [512, 512]

[numerics]
{dt_line}max_steps = 30000
"""

    env = os.environ.copy()
    brew_lib = Path("/opt/homebrew/lib")
    icd      = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    if icd.exists() and "VK_ICD_FILENAMES" not in env:
        env["VK_ICD_FILENAMES"] = str(icd)
    if brew_lib.exists():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        if str(brew_lib) not in existing:
            env["DYLD_LIBRARY_PATH"] = (str(brew_lib) + ":" + existing).rstrip(":")

    import shutil
    results_ke = {}
    results_y  = {}

    for label, fixed in [("fixed", True), ("adaptive", False)]:
        deck_path = VALDATA / f"t15_{label}.toml"
        deck_path.write_text(make_deck(fixed))
        out = VALOUT / f"t15_{label}"
        if out.exists():
            shutil.rmtree(out)
        cmd = [str(BIN), "run", str(deck_path), "-o", str(out)]
        try:
            r = subprocess.run(cmd, cwd=VALDATA, capture_output=True,
                               text=True, timeout=120, env=env)
        except subprocess.TimeoutExpired:
            REPORT["test15_adaptive_dt"] = {"pass": False, "error": f"timeout ({label})"}
            return False
        if r.returncode != 0:
            print(f"   tracer error ({label}): {r.stderr[-300:]}")
            REPORT["test15_adaptive_dt"] = {"pass": False, "error": f"simulation failed ({label})"}
            return False

        hits_bin = out / "counts" / "hits.bin"
        if not hits_bin.exists():
            print(f"   no hits.bin ({label})")
            REPORT["test15_adaptive_dt"] = {"pass": False, "error": f"no hits ({label})"}
            return False
        raw = hits_bin.read_bytes()
        hits_data = np.frombuffer(raw, dtype="<f4", offset=4).reshape(-1, 3)
        results_ke[label] = hits_data[:, 2]
        results_y[label]  = hits_data[:, 0]  # local-y hit position

    n_fixed    = len(results_ke["fixed"])
    n_adaptive = len(results_ke["adaptive"])
    mean_y_fixed    = float(np.mean(results_y["fixed"]))
    mean_y_adaptive = float(np.mean(results_y["adaptive"]))
    rel_std_fixed    = float(np.std(results_ke["fixed"])  / np.mean(results_ke["fixed"]))
    rel_std_adaptive = float(np.std(results_ke["adaptive"]) / np.mean(results_ke["adaptive"]))

    hits_ok   = n_fixed >= 10000 and n_adaptive >= 10000
    pos_ok    = abs(mean_y_fixed - mean_y_adaptive) < 1.0  # within 1 mm
    energy_ok = rel_std_fixed < 1e-4 and rel_std_adaptive < 1e-4

    ok = hits_ok and pos_ok and energy_ok
    print(f"   fixed:    hits={n_fixed}, mean_y={mean_y_fixed:.3f} mm, ke_rel_std={rel_std_fixed:.2e}")
    print(f"   adaptive: hits={n_adaptive}, mean_y={mean_y_adaptive:.3f} mm, ke_rel_std={rel_std_adaptive:.2e}")
    print(f"   Δmean_y = {abs(mean_y_fixed - mean_y_adaptive):.4f} mm  (tol 1.0 mm)")
    if not hits_ok:
        print(f"   FAIL: insufficient hits")
    if not pos_ok:
        print(f"   FAIL: hit positions disagree by {abs(mean_y_fixed - mean_y_adaptive):.3f} mm")
    if not energy_ok:
        print(f"   FAIL: energy not conserved")

    REPORT["test15_adaptive_dt"] = {
        "pass": ok,
        "n_hits_fixed": n_fixed,
        "n_hits_adaptive": n_adaptive,
        "mean_y_fixed_mm": round(mean_y_fixed, 4),
        "mean_y_adaptive_mm": round(mean_y_adaptive, 4),
        "delta_mean_y_mm": round(abs(mean_y_fixed - mean_y_adaptive), 4),
        "ke_rel_std_fixed": float(f"{rel_std_fixed:.3e}"),
        "ke_rel_std_adaptive": float(f"{rel_std_adaptive:.3e}"),
    }
    return ok


# ── test 16: Bethe-Bloch energy loss ─────────────────────────────────────────

def write_dens(path, rho_arr, bounds):
    """Write a .dens binary file.  rho_arr: (nx,ny,nz) float32 array [g/cm³]."""
    nx, ny, nz = rho_arr.shape
    xmn, xmx, ymn, ymx, zmn, zmx = bounds
    with open(path, "wb") as f:
        f.write(b"DENS")
        f.write(struct.pack("<I", 1))              # version
        f.write(struct.pack("<III", nx, ny, nz))
        f.write(struct.pack("<6f", xmn, xmx, ymn, ymx, zmn, zmx))
        f.write(b"\x00" * (64 - 4 - 4 - 12 - 24))  # padding to 64 bytes
        f.write(rho_arr.astype("<f4").tobytes())


def bethe_bloch_water_mev_cm2_g(ke_mev):
    """Bethe-Bloch mass stopping power [MeV cm²/g] for proton in water."""
    K = 0.307075
    ME_C2 = 0.51099895
    MP_C2 = 938.272046
    I_MEV = 75.0e-6  # water mean excitation energy

    gamma = 1.0 + ke_mev / MP_C2
    beta2 = 1.0 - 1.0 / gamma**2
    tmax = 2 * ME_C2 * beta2 * gamma**2 / (1 + 2*gamma*ME_C2/MP_C2 + (ME_C2/MP_C2)**2)
    arg = 2 * ME_C2 * beta2 * gamma**2 * tmax / I_MEV**2
    bracket = 0.5 * np.log(arg) - beta2
    if bracket <= 0:
        return 0.0
    return K * 0.5551 / beta2 * bracket   # Z/A_water = 0.5551


def analytic_ke_loss_water(ke0_mev, thickness_m):
    """CSDA energy loss [MeV] for a proton in water (ρ=1 g/cm³) over thickness_m [m]."""
    # Integrate dE/dx via simple trapezoidal quadrature from ke0 downward
    n_steps = 1000
    ke = ke0_mev
    dx_cm = thickness_m * 100.0 / n_steps  # path element [cm]
    for _ in range(n_steps):
        dedx = bethe_bloch_water_mev_cm2_g(ke)  # MeV cm²/g; × ρ=1 g/cm³ = MeV/cm
        dE = dedx * dx_cm
        ke -= dE
        if ke <= 0.001:
            return ke0_mev - 0.001
    return ke0_mev - ke


def test16_bethe_bloch():
    """
    Proton beam (pencil) through a uniform water slab (ρ = 1 g/cm³, 5 mm thick).
    Checks that the GPU simulation energy loss matches the analytic Bethe-Bloch
    integral to within ±5%.  Also verifies no hit has more energy than the input
    (energy is never gained).
    """
    print("Test 16: Bethe-Bloch energy loss in water slab")
    VALDATA.mkdir(parents=True, exist_ok=True)

    SLAB_THICKNESS_M = 0.001   # 1 mm water slab (14.7 MeV range ≈ 2.4 mm → protons pass through)
    RHO_WATER = 1.0            # g/cm³
    ENERGY_MEV = 14.7
    N_PARTICLES = 10_000

    # Slab spans x ∈ [0, SLAB_THICKNESS_M]; y,z ∈ [-0.05, 0.05 m]
    slab_bounds = (0.0, SLAB_THICKNESS_M, -0.05, 0.05, -0.05, 0.05)

    # Build 4×16×16 density grid: uniform ρ = 1 g/cm³ inside slab
    dens = np.full((4, 16, 16), RHO_WATER, dtype=np.float32)
    dens_path = VALDATA / "t16_water_slab.dens"
    write_dens(dens_path, dens, slab_bounds)

    # Zero B-field spanning a broader region
    bfld_path = VALDATA / "t16_zero.bfld"
    B = np.zeros((4, 8, 8, 3), dtype=np.float32)
    E = np.zeros_like(B)
    write_bfld(bfld_path, B, E, (-0.01, 0.16, -0.05, 0.05, -0.05, 0.05))

    deck = f"""\
[field]
path = "t16_zero.bfld"
scale_B = 0.0
scale_E = 0.0

[density]
path = "t16_water_slab.dens"
material = "water"

[source]
type = "pencil"
n_particles = {N_PARTICLES}
energy_MeV = {ENERGY_MEV}
position_mm = [-80.0, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 200.0
height_mm = 200.0
pixels = [256, 256]

[numerics]
dt_ps = 0.5
max_steps = 30000
"""
    deck_path = VALDATA / "t16_water_slab.toml"
    deck_path.write_text(deck)

    out = VALOUT / "t16_bethe_bloch"
    import shutil
    if out.exists():
        shutil.rmtree(out)

    env = os.environ.copy()
    brew_lib = Path("/opt/homebrew/lib")
    icd = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    if icd.exists() and "VK_ICD_FILENAMES" not in env:
        env["VK_ICD_FILENAMES"] = str(icd)
    if brew_lib.exists():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        if str(brew_lib) not in existing:
            env["DYLD_LIBRARY_PATH"] = (str(brew_lib) + ":" + existing).rstrip(":")

    cmd = [str(BIN), "run", str(deck_path), "-o", str(out)]
    try:
        result = subprocess.run(cmd, cwd=VALDATA, capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        REPORT["test16_bethe_bloch"] = {"pass": False, "error": "timeout"}
        return False
    if result.returncode != 0:
        print(f"   tracer error: {result.stderr[-500:]}")
        REPORT["test16_bethe_bloch"] = {"pass": False, "error": "simulation failed"}
        return False

    hits_bin = out / "counts" / "hits.bin"
    if not hits_bin.exists():
        print("   no hits.bin found")
        REPORT["test16_bethe_bloch"] = {"pass": False, "error": "no hits.bin"}
        return False
    raw = hits_bin.read_bytes()
    if len(raw) < 4:
        print("   hits.bin too short")
        REPORT["test16_bethe_bloch"] = {"pass": False, "error": "no hits"}
        return False

    n_rec = struct.unpack_from("<I", raw, 0)[0]
    hits_data = np.frombuffer(raw, dtype="<f4", offset=4).reshape(-1, 3)
    ke_arr = hits_data[:, 2]

    # Analytic expected energy loss
    expected_loss = analytic_ke_loss_water(ENERGY_MEV, SLAB_THICKNESS_M)
    expected_ke_exit = ENERGY_MEV - expected_loss

    mean_ke  = float(np.mean(ke_arr))
    max_ke   = float(np.max(ke_arr))
    rel_err  = abs(mean_ke - expected_ke_exit) / expected_ke_exit

    tol_rel  = 0.05   # ±5% tolerance on mean exit KE

    ok1 = rel_err <= tol_rel
    ok2 = max_ke <= ENERGY_MEV + 0.01   # no particle gains energy (allow 10 keV rounding)
    ok  = ok1 and ok2

    print(f"   hits = {len(ke_arr)},  mean KE = {mean_ke:.4f} MeV  (expected {expected_ke_exit:.4f} MeV)")
    print(f"   energy loss: simulated = {ENERGY_MEV - mean_ke:.3f} MeV,  analytic = {expected_loss:.3f} MeV")
    print(f"   relative error = {rel_err:.3f}  (tol {tol_rel})")
    if not ok1:
        print(f"   FAIL: relative error {rel_err:.3f} > {tol_rel}")
    if not ok2:
        print(f"   FAIL: max KE {max_ke:.4f} > input {ENERGY_MEV} MeV (energy gained!)")

    REPORT["test16_bethe_bloch"] = {
        "pass": ok,
        "n_hits": int(len(ke_arr)),
        "mean_ke_sim_MeV": round(mean_ke, 4),
        "expected_ke_MeV": round(expected_ke_exit, 4),
        "loss_sim_MeV": round(ENERGY_MEV - mean_ke, 4),
        "loss_analytic_MeV": round(expected_loss, 4),
        "relative_error": float(f"{rel_err:.4f}"),
        "tolerance": tol_rel,
    }
    return ok


# ── helpers for run-subcommand tests ─────────────────────────────────────────

def _vulkan_env():
    """Return os.environ copy with MoltenVK ICD configured if present."""
    env = os.environ.copy()
    brew_lib = Path("/opt/homebrew/lib")
    icd      = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    if icd.exists() and "VK_ICD_FILENAMES" not in env:
        env["VK_ICD_FILENAMES"] = str(icd)
    if brew_lib.exists():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        if str(brew_lib) not in existing:
            env["DYLD_LIBRARY_PATH"] = (str(brew_lib) + ":" + existing).rstrip(":")
    return env


def run_deck(deck_path, out_dir, timeout=120):
    """Run the 'run' subcommand with a TOML deck; return True on success."""
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    env = _vulkan_env()
    cmd = [str(BIN), "run", str(deck_path), "-o", str(out_dir)]
    try:
        r = subprocess.run(cmd, cwd=VALDATA, capture_output=True,
                           text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print("   TIMEOUT")
        return False
    for line in (r.stdout + r.stderr).splitlines():
        upper = line.upper()
        if any(tok in upper for tok in ("ERROR", "WARN", "COMPLETE", "EXPORTED", "HITS")):
            print("   ", line.strip())
    return r.returncode == 0


def read_hits_bin(out_dir):
    """Read counts/hits.bin → ndarray (N,3) of (y_mm, z_mm, ke_MeV)."""
    hits_bin = out_dir / "counts" / "hits.bin"
    if not hits_bin.exists():
        return np.zeros((0, 3), dtype=np.float32)
    raw = hits_bin.read_bytes()
    if len(raw) < 4:
        return np.zeros((0, 3), dtype=np.float32)
    return np.frombuffer(raw, dtype="<f4", offset=4).reshape(-1, 3)


# ── test 17: analytic straight-line detector hit ──────────────────────────────

def test17_analytic_hit():
    """
    Off-axis pencil beam (zero field): GPU hit must match the analytic
    ray–plane intersection to within 0.1 mm.

    Source at (−100, +20, −15) mm, direction along (100, 15, −8) normalised.
    Detector at (110, 0, 0) mm, normal +x, up +y.

    Analytic:  r_hit = src + t*d  where t = n·(det−src)/(n·d).
    Expected:  y_local = +51.5 mm,  z_local = −31.8 mm.

    Tests: source position, detector plane, Gram-Schmidt basis, ray–plane
           intersection, and per-hit hits.bin coordinate accuracy.
    """
    print("Test 17: Analytic straight-line hit  (ray–plane intersection, 0.1 mm)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    src_pos = np.array([-100.0, 20.0, -15.0])   # mm
    aim_at  = np.array([   0.0, 35.0, -23.0])   # mm  → direction (100,15,−8)

    d_raw = aim_at - src_pos                     # (100, 15, -8)
    d     = d_raw / np.linalg.norm(d_raw)

    det_center = np.array([110.0, 0.0, 0.0])
    det_normal = np.array([  1.0, 0.0, 0.0])
    det_up     = np.array([  0.0, 1.0, 0.0])

    t_hit  = np.dot(det_normal, det_center - src_pos) / np.dot(det_normal, d)
    r_hit  = src_pos + t_hit * d
    offset = r_hit - det_center

    u_y = det_up - np.dot(det_up, det_normal) * det_normal
    u_y /= np.linalg.norm(u_y)
    v_z = np.cross(det_normal, u_y)

    y_analytic = float(np.dot(offset, u_y))
    z_analytic = float(np.dot(offset, v_z))

    B = np.zeros((4, 4, 4, 3), dtype=np.float32)
    E = np.zeros_like(B)
    write_bfld(VALDATA / "t17_zero.bfld", B, E,
               (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06))

    deck = f"""\
[field]
path = "t17_zero.bfld"
scale_B = 0.0
scale_E = 0.0

[source]
type = "pencil"
n_particles = 10000
energy_MeV = 14.7
position_mm = {src_pos.tolist()}
aim_at_mm = {aim_at.tolist()}

[detector]
center_mm = {det_center.tolist()}
normal = {det_normal.tolist()}
up = {det_up.tolist()}
width_mm = 500.0
height_mm = 500.0
pixels = [256, 256]

[numerics]
dt_ps = 1.0
max_steps = 20000
"""
    (VALDATA / "t17_analytic_hit.toml").write_text(deck)

    out = VALOUT / "t17_analytic_hit"
    if not run_deck(VALDATA / "t17_analytic_hit.toml", out):
        REPORT["test17_analytic_hit"] = {"pass": False, "error": "simulation failed"}
        return False

    data = read_hits_bin(out)
    if len(data) == 0:
        print("   no hits")
        REPORT["test17_analytic_hit"] = {"pass": False, "error": "no hits"}
        return False

    y_gpu  = float(np.mean(data[:, 0]))
    z_gpu  = float(np.mean(data[:, 1]))
    std_y  = float(np.std(data[:, 0]))
    std_z  = float(np.std(data[:, 1]))

    tol = 0.1   # mm
    ok_y   = abs(y_gpu - y_analytic) < tol
    ok_z   = abs(z_gpu - z_analytic) < tol
    ok_std = std_y < tol and std_z < tol
    ok     = ok_y and ok_z and ok_std

    print(f"   hits = {len(data)}")
    print(f"   y: GPU = {y_gpu:+.4f} mm,  analytic = {y_analytic:+.4f} mm,  "
          f"diff = {y_gpu - y_analytic:+.4f} mm")
    print(f"   z: GPU = {z_gpu:+.4f} mm,  analytic = {z_analytic:+.4f} mm,  "
          f"diff = {z_gpu - z_analytic:+.4f} mm")
    print(f"   std_y = {std_y:.4f} mm,  std_z = {std_z:.4f} mm  (pencil → ≈ 0)")
    if not ok_y:
        print(f"   FAIL: y diff {y_gpu - y_analytic:+.4f} mm exceeds ±{tol} mm")
    if not ok_z:
        print(f"   FAIL: z diff {z_gpu - z_analytic:+.4f} mm exceeds ±{tol} mm")
    if not ok_std:
        print(f"   FAIL: std > {tol} mm for pencil beam")

    REPORT["test17_analytic_hit"] = {
        "pass": ok,
        "n_hits": len(data),
        "y_gpu_mm": round(y_gpu, 4), "y_analytic_mm": round(y_analytic, 4),
        "y_diff_mm": round(y_gpu - y_analytic, 4),
        "z_gpu_mm": round(z_gpu, 4), "z_analytic_mm": round(z_analytic, 4),
        "z_diff_mm": round(z_gpu - z_analytic, 4),
        "std_y_mm": round(std_y, 5), "std_z_mm": round(std_z, 5),
        "tolerance_mm": tol,
    }
    return ok


# ── test 18: analytic Larmor radius ───────────────────────────────────────────

def test18_larmor_radius():
    """
    Pencil beam through uniform Bz = 0.1 T.  Exact cycloid formula:
      r_L = p/(qB),  θ_exit = arcsin(L/r_L)
      y_det = r_L(cos θ_exit − 1) + tan(θ_exit)(x_det − x_exit)

    GPU mean deflection must agree with the analytic value to within 1 %.

    Tests: relativistic Boris integrator accuracy against a closed-form helix.
    """
    print("Test 18: Larmor radius  (analytic cycloid vs GPU Boris, tol 1 %)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    E_MEV = 14.7
    B_T   = 0.1
    C     = 2.99792458e8
    MP_C2 = 938.272046
    Q     = 1.602176634e-19
    MP_KG = MP_C2 * 1.602176634e-13 / C**2

    gamma = 1.0 + E_MEV / MP_C2
    beta  = float(np.sqrt(1.0 - 1.0 / gamma**2))
    v     = beta * C
    p_SI  = gamma * MP_KG * v
    r_L   = p_SI / (Q * B_T)             # Larmor radius [m]

    # Field: x ∈ [−0.06, +0.06] m  →  L = 0.12 m
    L             = 0.12
    x_field_exit  = 0.06
    x_det         = 0.11

    s       = L / r_L
    y_exit  = r_L * (np.sqrt(1.0 - s**2) - 1.0)   # m, negative
    vy_vx   = -s / np.sqrt(1.0 - s**2)
    y_det_m = y_exit + vy_vx * (x_det - x_field_exit)
    y_det_mm = y_det_m * 1000.0

    B_arr = np.zeros((16, 16, 16, 3), dtype=np.float32)
    B_arr[..., 2] = float(B_T)
    E_arr = np.zeros_like(B_arr)
    write_bfld(VALDATA / "t18_Bz.bfld", B_arr, E_arr,
               (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06))

    deck = f"""\
[field]
path = "t18_Bz.bfld"
scale_B = 1.0
scale_E = 0.0

[source]
type = "pencil"
n_particles = 10000
energy_MeV = {E_MEV}
position_mm = [-100.0, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [256, 256]

[numerics]
dt_ps = 0.5
max_steps = 20000
"""
    (VALDATA / "t18_larmor.toml").write_text(deck)

    out = VALOUT / "t18_larmor"
    if not run_deck(VALDATA / "t18_larmor.toml", out):
        REPORT["test18_larmor_radius"] = {"pass": False, "error": "simulation failed"}
        return False

    data = read_hits_bin(out)
    if len(data) == 0:
        print("   no hits")
        REPORT["test18_larmor_radius"] = {"pass": False, "error": "no hits"}
        return False

    y_gpu   = float(np.mean(data[:, 0]))
    rel_err = abs(y_gpu - y_det_mm) / abs(y_det_mm)
    tol_rel = 0.01

    ok = rel_err < tol_rel
    print(f"   hits = {len(data)},  r_L = {r_L:.4f} m")
    print(f"   analytic y_det = {y_det_mm:.4f} mm,  GPU = {y_gpu:.4f} mm,  "
          f"rel_err = {rel_err:.5f}  (tol {tol_rel})")
    if not ok:
        print(f"   FAIL: rel_err {rel_err:.5f} > {tol_rel}")

    REPORT["test18_larmor_radius"] = {
        "pass": ok,
        "larmor_radius_m": round(r_L, 5),
        "analytic_y_mm": round(y_det_mm, 4),
        "gpu_y_mm": round(y_gpu, 4),
        "abs_err_mm": round(abs(y_gpu - y_det_mm), 4),
        "relative_error": round(rel_err, 6),
        "tolerance": tol_rel,
    }
    return ok


# ── test 19: E × B velocity selector ─────────────────────────────────────────

def test19_exb_velocity_selector():
    """
    Velocity-selector force balance: with E_y = v_beam × B_z, the electric and
    magnetic transverse forces cancel exactly and the proton goes straight.

    Two runs using the same field file (Bz = 0.1 T, Ey = v_beam × Bz):
      scale_E = 0  (B only)  → mean_y ≈ −2.4 mm  (large deflection)
      scale_E = 1  (B + E)   → mean_y ≈  0.0 mm  (force balance)

    Tests: correct sign and magnitude of both E and B forces, and vector sum.
    """
    print("Test 19: E×B velocity selector  (E_y = v_beam × B_z → straight beam)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    E_MEV = 14.7
    B_T   = 0.1
    C     = 2.99792458e8
    MP_C2 = 938.272046
    gamma  = 1.0 + E_MEV / MP_C2
    v_beam = float(np.sqrt(1.0 - 1.0 / gamma**2)) * C
    E_y    = v_beam * B_T                 # [V/m] — cancels magnetic force

    B_arr = np.zeros((16, 16, 16, 3), dtype=np.float32)
    B_arr[..., 2] = float(B_T)
    E_arr = np.zeros((16, 16, 16, 3), dtype=np.float32)
    E_arr[..., 1] = float(E_y)
    write_bfld(VALDATA / "t19_ExB.bfld", B_arr, E_arr,
               (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06))

    template = """\
[field]
path = "t19_ExB.bfld"
scale_B = 1.0
scale_E = {se}

[source]
type = "pencil"
n_particles = 20000
energy_MeV = {emev}
position_mm = [-100.0, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [256, 256]

[numerics]
dt_ps = 0.5
max_steps = 20000
"""
    runs = {"B_only": 0.0, "B_plus_E": 1.0}
    mean_y = {}
    for label, se in runs.items():
        deck_path = VALDATA / f"t19_{label}.toml"
        deck_path.write_text(template.format(se=se, emev=E_MEV))
        out = VALOUT / f"t19_{label}"
        if not run_deck(deck_path, out):
            REPORT["test19_exb_velocity_selector"] = {"pass": False, "error": f"sim failed ({label})"}
            return False
        data = read_hits_bin(out)
        if len(data) == 0:
            REPORT["test19_exb_velocity_selector"] = {"pass": False, "error": f"no hits ({label})"}
            return False
        mean_y[label] = float(np.mean(data[:, 0]))

    y_B   = mean_y["B_only"]
    y_bal = mean_y["B_plus_E"]

    ok_B       = y_B   < -1.0              # B-only must deflect in −y
    ok_balance = abs(y_bal) < 1.0          # force-balance must stay near zero
    ok_contrast = (y_B - y_bal) < -1.0    # runs must differ
    ok = ok_B and ok_balance and ok_contrast

    print(f"   B-only     mean_y = {y_B:+.3f} mm  (expect < −1.0 mm)")
    print(f"   B+E bal    mean_y = {y_bal:+.3f} mm  (expect |y| < 1.0 mm)")
    if not ok_B:
        print(f"   FAIL: B-only not deflected (y = {y_B:+.3f})")
    if not ok_balance:
        print(f"   FAIL: force balance broken (|y| = {abs(y_bal):.3f} mm > 1 mm)")
    if not ok_contrast:
        print(f"   FAIL: B-only and B+E too similar")

    REPORT["test19_exb_velocity_selector"] = {
        "pass": ok,
        "Bz_T": B_T,
        "Ey_MV_m": round(E_y / 1e6, 4),
        "v_beam_Mm_s": round(v_beam / 1e6, 3),
        "mean_y_B_only_mm": round(y_B, 4),
        "mean_y_balance_mm": round(y_bal, 4),
        "contrast_mm": round(y_B - y_bal, 4),
    }
    return ok


# ── test 20: hits.bin rebinning consistency ────────────────────────────────────

def test20_hits_bin_rebinning():
    """
    Re-bin per-hit (y_mm, z_mm) records in Python using the GPU pixel formula:
      col = int((y_mm + W/2) / W * 1024),  row = int((z_mm + H/2) / H * 1024)

    Compare with raw_counts.bin (1024 × 1024 u32).

    Checks:
      1. Total rebinned count equals total raw count (no hits lost)
      2. ≥ 99.5 % of occupied pixels agree within ±1 count
         (allows for float32 rounding at pixel boundaries)

    Tests: hits.bin position accuracy, coordinate convention, pixel mapping.
    """
    print("Test 20: hits.bin rebinning consistency  (Python histogram ↔ GPU image)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    W_MM = 200.0
    H_MM = 200.0
    RES  = 1024
    N    = 50_000

    B = np.zeros((4, 4, 4, 3), dtype=np.float32)
    E = np.zeros_like(B)
    write_bfld(VALDATA / "t20_zero.bfld", B, E,
               (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06))

    deck = f"""\
[field]
path = "t20_zero.bfld"
scale_B = 0.0
scale_E = 0.0

[source]
type = "disk"
n_particles = {N}
energy_MeV = 14.7
center_mm = [-100.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
radius_um = 40000.0
cone_half_angle_deg = 0.0

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = {W_MM}
height_mm = {H_MM}
pixels = [256, 256]

[numerics]
dt_ps = 1.0
max_steps = 20000
"""
    (VALDATA / "t20_rebinning.toml").write_text(deck)

    out = VALOUT / "t20_rebinning"
    if not run_deck(VALDATA / "t20_rebinning.toml", out):
        REPORT["test20_hits_bin_rebinning"] = {"pass": False, "error": "sim failed"}
        return False

    data = read_hits_bin(out)
    raw_bin = out / "counts" / "raw_counts.bin"
    if len(data) == 0 or not raw_bin.exists():
        print("   missing hits or counts")
        REPORT["test20_hits_bin_rebinning"] = {"pass": False, "error": "missing output"}
        return False

    raw = np.frombuffer(raw_bin.read_bytes(), dtype="<u4").reshape(RES, RES)

    y_mm = data[:, 0]
    z_mm = data[:, 1]
    col  = np.clip(((y_mm + W_MM / 2.0) / W_MM * RES).astype(np.int32), 0, RES - 1)
    row  = np.clip(((z_mm + H_MM / 2.0) / H_MM * RES).astype(np.int32), 0, RES - 1)

    rebin = np.zeros((RES, RES), dtype=np.int64)
    np.add.at(rebin, (row, col), 1)

    total_raw   = int(raw.sum())
    total_rebin = int(rebin.sum())
    diff        = np.abs(rebin.astype(np.int64) - raw.astype(np.int64))
    occupied    = (raw > 0) | (rebin > 0)
    n_occupied  = int(occupied.sum())
    frac_ok     = float((diff[occupied] <= 1).sum()) / max(n_occupied, 1)

    total_ok = (total_raw == total_rebin)
    agree_ok = (frac_ok >= 0.995)
    ok = total_ok and agree_ok

    print(f"   hits = {len(data)},  total_raw = {total_raw},  total_rebin = {total_rebin}")
    print(f"   occupied pixels = {n_occupied},  "
          f"agree within ±1 = {frac_ok:.4f}  (tol 0.995)")
    if not total_ok:
        print(f"   FAIL: total count mismatch ({total_raw} raw vs {total_rebin} rebin)")
    if not agree_ok:
        print(f"   FAIL: pixel agreement {frac_ok:.4f} < 0.995")

    REPORT["test20_hits_bin_rebinning"] = {
        "pass": ok,
        "n_hits": len(data),
        "total_raw": total_raw,
        "total_rebin": total_rebin,
        "n_occupied_pixels": n_occupied,
        "frac_within_1_count": round(frac_ok, 6),
    }
    return ok


# ── test 21: geometry invariance ─────────────────────────────────────────────

def test21_geometry_invariance():
    """
    Run the same physics (uniform Bz = 0.1 T, parallel beam) in two orientations
    that are 90° rotations of each other.  The deflection magnitude in
    detector-local coordinates must be identical.

    Case A: beam +x → detector at (110, 0, 0) mm, up = (0, 1, 0)
      Bz deflects in −y_world  → local_y < 0

    Case B: beam +y → detector at (0, 110, 0) mm, up = (−1, 0, 0)
      Bz deflects in +x_world  → u_y = (−1,0,0)  → local_y < 0  (same sign)

    Assertion: |mean_y_A − mean_y_B| < 0.5 mm
    """
    print("Test 21: Geometry invariance  (90° world rotation, same local deflection)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    B_arr = np.zeros((16, 16, 16, 3), dtype=np.float32)
    B_arr[..., 2] = 0.1
    E_arr = np.zeros_like(B_arr)
    write_bfld(VALDATA / "t21_Bz.bfld", B_arr, E_arr,
               (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06))

    decks = {
        "A_beam_x": """\
[field]
path = "t21_Bz.bfld"
scale_B = 1.0
scale_E = 0.0

[source]
type = "parallel"
n_particles = 20000
energy_MeV = 14.7
beam_radius_mm = 30.0
source_distance_mm = 100.0
direction = [1.0, 0.0, 0.0]

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [256, 256]

[numerics]
dt_ps = 0.5
max_steps = 20000
""",
        "B_beam_y": """\
[field]
path = "t21_Bz.bfld"
scale_B = 1.0
scale_E = 0.0

[source]
type = "parallel"
n_particles = 20000
energy_MeV = 14.7
beam_radius_mm = 30.0
source_distance_mm = 100.0
direction = [0.0, 1.0, 0.0]

[detector]
center_mm = [0.0, 110.0, 0.0]
normal = [0.0, 1.0, 0.0]
up = [-1.0, 0.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [256, 256]

[numerics]
dt_ps = 0.5
max_steps = 20000
""",
    }

    mean_y = {}
    for label, deck_txt in decks.items():
        deck_path = VALDATA / f"t21_{label}.toml"
        deck_path.write_text(deck_txt)
        out = VALOUT / f"t21_{label}"
        if not run_deck(deck_path, out):
            REPORT["test21_geometry_invariance"] = {"pass": False, "error": f"sim failed ({label})"}
            return False
        data = read_hits_bin(out)
        if len(data) == 0:
            REPORT["test21_geometry_invariance"] = {"pass": False, "error": f"no hits ({label})"}
            return False
        mean_y[label] = float(np.mean(data[:, 0]))

    y_A   = mean_y["A_beam_x"]
    y_B   = mean_y["B_beam_y"]
    diff  = abs(y_A - y_B)
    tol   = 0.5   # mm

    sign_ok = (y_A < 0) and (y_B < 0)
    match_ok = diff < tol
    ok = sign_ok and match_ok

    print(f"   Case A (+x beam):  mean_y = {y_A:+.4f} mm")
    print(f"   Case B (+y beam):  mean_y = {y_B:+.4f} mm")
    print(f"   |difference| = {diff:.4f} mm  (tol {tol} mm)")
    if not sign_ok:
        print(f"   FAIL: signs differ  (both should be negative in detector-local frame)")
    if not match_ok:
        print(f"   FAIL: |Δ| = {diff:.4f} mm > {tol} mm — rotation changed the physics")

    REPORT["test21_geometry_invariance"] = {
        "pass": ok,
        "mean_y_beam_x_mm": round(y_A, 4),
        "mean_y_beam_y_mm": round(y_B, 4),
        "diff_mm": round(diff, 4),
        "tolerance_mm": tol,
    }
    return ok


# ── test 22: field compositing linearity ──────────────────────────────────────

def test22_field_compositing_linearity():
    """
    Orthogonal B components deflect in orthogonal detector channels; superimposing
    them should leave each channel unaffected by the other (to first order,
    L/r_L ≈ 0.02 here, so cross-coupling ~ 0.02 % — well within 5 % tolerance).

    Bz = 0.1 T deflects in −y (F = q v_x Bz → −y).
    By = 0.1 T deflects in +z (F = q v_x By → +z).

    Three runs (pencil, +x beam):
      A: zero primary + Bz extra   →  y_A < 0,  z_A ≈ 0
      B: zero primary + By extra   →  y_B ≈ 0,  z_B > 0
      C: zero primary + Bz + By    →  y_C ≈ y_A,  z_C ≈ z_B  (superposition)
    """
    print("Test 22: Field compositing linearity  (orthogonal B superposition)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)
    B_zero = np.zeros((2, 2, 2, 3), dtype=np.float32)
    E_zero = np.zeros_like(B_zero)
    write_bfld(VALDATA / "t22_zero.bfld", B_zero, E_zero, bounds)

    B_bz = np.zeros((16, 16, 16, 3), dtype=np.float32); B_bz[..., 2] = 0.1
    B_by = np.zeros((16, 16, 16, 3), dtype=np.float32); B_by[..., 1] = 0.1
    Ez   = np.zeros((16, 16, 16, 3), dtype=np.float32)
    write_bfld(VALDATA / "t22_Bz.bfld", B_bz, Ez, bounds)
    write_bfld(VALDATA / "t22_By.bfld", B_by, Ez, bounds)

    common = """\
[source]
type = "pencil"
n_particles = 10000
energy_MeV = 14.7
position_mm = [-100.0, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [256, 256]

[numerics]
dt_ps = 0.5
max_steps = 20000
"""
    decks = {
        "A_Bz": '[field]\npath = "t22_zero.bfld"\nscale_B = 0.0\n\n'
                '[[field.extra_b]]\npath = "t22_Bz.bfld"\nscale_B = 1.0\n\n' + common,
        "B_By": '[field]\npath = "t22_zero.bfld"\nscale_B = 0.0\n\n'
                '[[field.extra_b]]\npath = "t22_By.bfld"\nscale_B = 1.0\n\n' + common,
        "C_both": '[field]\npath = "t22_zero.bfld"\nscale_B = 0.0\n\n'
                  '[[field.extra_b]]\npath = "t22_Bz.bfld"\nscale_B = 1.0\n\n'
                  '[[field.extra_b]]\npath = "t22_By.bfld"\nscale_B = 1.0\n\n' + common,
    }

    hits = {}
    for label, deck_txt in decks.items():
        deck_path = VALDATA / f"t22_{label}.toml"
        deck_path.write_text(deck_txt)
        out = VALOUT / f"t22_{label}"
        if not run_deck(deck_path, out):
            REPORT["test22_field_compositing_linearity"] = {"pass": False, "error": f"sim failed ({label})"}
            return False
        data = read_hits_bin(out)
        if len(data) == 0:
            REPORT["test22_field_compositing_linearity"] = {"pass": False, "error": f"no hits ({label})"}
            return False
        hits[label] = (float(np.mean(data[:, 0])), float(np.mean(data[:, 1])))

    y_A, z_A = hits["A_Bz"]
    y_B, z_B = hits["B_By"]
    y_C, z_C = hits["C_both"]

    y_rel = abs(y_C - y_A) / max(abs(y_A), 1e-3)
    z_rel = abs(z_C - z_B) / max(abs(z_B), 1e-3)
    tol = 0.05

    ok_sign_A = y_A < 0
    ok_sign_B = z_B > 0
    ok_y  = y_rel < tol
    ok_z  = z_rel < tol
    ok = ok_sign_A and ok_sign_B and ok_y and ok_z

    print(f"   A (Bz only):   y = {y_A:+.3f},  z = {z_A:+.3f} mm")
    print(f"   B (By only):   y = {y_B:+.3f},  z = {z_B:+.3f} mm")
    print(f"   C (Bz + By):   y = {y_C:+.3f},  z = {z_C:+.3f} mm")
    print(f"   y_rel_err = {y_rel:.4f}  (C vs A,  tol {tol})")
    print(f"   z_rel_err = {z_rel:.4f}  (C vs B,  tol {tol})")
    if not ok_sign_A:
        print(f"   FAIL: Bz-only should deflect in −y")
    if not ok_sign_B:
        print(f"   FAIL: By-only should deflect in +z")
    if not ok_y:
        print(f"   FAIL: y channel rel_err {y_rel:.4f} > {tol}")
    if not ok_z:
        print(f"   FAIL: z channel rel_err {z_rel:.4f} > {tol}")

    REPORT["test22_field_compositing_linearity"] = {
        "pass": ok,
        "A_y_mm": round(y_A, 4), "A_z_mm": round(z_A, 4),
        "B_y_mm": round(y_B, 4), "B_z_mm": round(z_B, 4),
        "C_y_mm": round(y_C, 4), "C_z_mm": round(z_C, 4),
        "y_rel_err": round(y_rel, 5),
        "z_rel_err": round(z_rel, 5),
        "tolerance": tol,
    }
    return ok


# ── test 23: density scaling  ΔE ∝ ρ L ───────────────────────────────────────

def test23_density_scaling():
    """
    Three water-equivalent slabs with matched and mismatched ρ × L (column depth):
      A: 1 mm, ρ = 1 g/cm³   →  ρL = 0.1 g/cm²
      B: 2 mm, ρ = 1 g/cm³   →  ρL = 0.2 g/cm²
      C: 1 mm, ρ = 2 g/cm³   →  ρL = 0.2 g/cm²

    ρL equivalence (range-energy theorem):  slabs B and C have the same column
    depth → same energy loss, regardless of how ρ is distributed along the path.
    This is EXACT in CSDA (Bethe-Bloch depends on Z/A and I, not density).

    Checks:
      1. Each slab's mean exit KE within 5 % of analytic_ke_loss_water(ρ→eq thickness)
      2. |ΔE_C − ΔE_B| / ΔE_B < 3 %  (ρL equivalence, should be exact)
    """
    print("Test 23: Density scaling  (ΔE ∝ ρL;  ρL equivalence B ≡ C)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    E_MEV = 14.7
    N     = 10_000

    B_arr = np.zeros((4, 8, 8, 3), dtype=np.float32)
    E_arr = np.zeros_like(B_arr)
    write_bfld(VALDATA / "t23_zero.bfld", B_arr, E_arr,
               (-0.01, 0.16, -0.05, 0.05, -0.05, 0.05))

    slabs = {
        "A_1mm_rho1": (0.001, 1.0),
        "B_2mm_rho1": (0.002, 1.0),
        "C_1mm_rho2": (0.001, 2.0),
    }

    results = {}
    for label, (L_m, rho) in slabs.items():
        dens_arr = np.full((4, 8, 8), rho, dtype=np.float32)
        dens_path = VALDATA / f"t23_{label}.dens"
        write_dens(dens_path, dens_arr, (0.0, L_m, -0.05, 0.05, -0.05, 0.05))

        # analytic via ρL equivalence: ΔE same as ρ=1 slab of thickness ρ×L
        eq_L = rho * L_m
        analytic_loss = analytic_ke_loss_water(E_MEV, eq_L)
        analytic_ke   = E_MEV - analytic_loss

        deck = f"""\
[field]
path = "t23_zero.bfld"
scale_B = 0.0
scale_E = 0.0

[density]
path = "{dens_path.name}"
material = "water"

[source]
type = "pencil"
n_particles = {N}
energy_MeV = {E_MEV}
position_mm = [-50.0, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 100.0
height_mm = 100.0
pixels = [128, 128]

[numerics]
dt_ps = 0.5
max_steps = 30000
"""
        deck_path = VALDATA / f"t23_{label}.toml"
        deck_path.write_text(deck)

        out = VALOUT / f"t23_{label}"
        if not run_deck(deck_path, out):
            REPORT["test23_density_scaling"] = {"pass": False, "error": f"sim failed ({label})"}
            return False
        data = read_hits_bin(out)
        if len(data) == 0:
            REPORT["test23_density_scaling"] = {"pass": False, "error": f"no hits ({label})"}
            return False

        mean_ke = float(np.mean(data[:, 2]))
        dE      = E_MEV - mean_ke
        rel_err = abs(mean_ke - analytic_ke) / analytic_ke
        results[label] = {"mean_ke": mean_ke, "dE": dE,
                          "analytic_ke": analytic_ke, "rel_err": rel_err}
        print(f"   {label}: mean_ke = {mean_ke:.4f} MeV  "
              f"(analytic {analytic_ke:.4f},  rel_err = {rel_err:.3f})")

    tol_ind   = 0.05
    tol_equiv = 0.03

    ok_A = results["A_1mm_rho1"]["rel_err"] < tol_ind
    ok_B = results["B_2mm_rho1"]["rel_err"] < tol_ind
    ok_C = results["C_1mm_rho2"]["rel_err"] < tol_ind

    dE_B     = results["B_2mm_rho1"]["dE"]
    dE_C     = results["C_1mm_rho2"]["dE"]
    eq_err   = abs(dE_C - dE_B) / max(dE_B, 0.001)
    ok_equiv = eq_err < tol_equiv

    ok = ok_A and ok_B and ok_C and ok_equiv

    print(f"   ρL equiv (B vs C): dE_B = {dE_B:.4f},  dE_C = {dE_C:.4f},  "
          f"rel_diff = {eq_err:.4f}  (tol {tol_equiv})")
    if not ok_equiv:
        print(f"   FAIL: ρL equivalence violated")

    REPORT["test23_density_scaling"] = {
        "pass": ok,
        "slab_A": {k: round(v, 5) if isinstance(v, float) else v
                   for k, v in results["A_1mm_rho1"].items()},
        "slab_B": {k: round(v, 5) if isinstance(v, float) else v
                   for k, v in results["B_2mm_rho1"].items()},
        "slab_C": {k: round(v, 5) if isinstance(v, float) else v
                   for k, v in results["C_1mm_rho2"].items()},
        "rhoL_equiv_rel_err": round(eq_err, 5),
        "tolerance_individual": tol_ind,
        "tolerance_equiv": tol_equiv,
    }
    return ok


# ── test 24: vacuum regression (no density block → energy conserved) ──────────

def test24_vacuum_regression():
    """
    TOML deck with no [density] block and Bz = 1 T.  The Bethe-Bloch GPU path
    must be a complete no-op: std(KE) / mean(KE) < 1e-4.

    This is the TOML-path analog of test4, confirming that Feature #5
    (stopping power) did not touch the energy-conserving vacuum default.
    """
    print("Test 24: Vacuum regression  (no [density] block → exact energy conservation)")
    VALDATA.mkdir(parents=True, exist_ok=True)

    B_arr = np.zeros((16, 16, 16, 3), dtype=np.float32)
    B_arr[..., 2] = 1.0
    E_arr = np.zeros_like(B_arr)
    write_bfld(VALDATA / "t24_Bz.bfld", B_arr, E_arr,
               (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06))

    deck = """\
[field]
path = "t24_Bz.bfld"
scale_B = 1.0
scale_E = 0.0

[source]
type = "parallel"
n_particles = 20000
energy_MeV = 14.7
beam_radius_mm = 30.0
source_distance_mm = 100.0

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [256, 256]

[numerics]
dt_ps = 1.0
max_steps = 20000
"""
    (VALDATA / "t24_vacuum_regression.toml").write_text(deck)

    out = VALOUT / "t24_vacuum_regression"
    if not run_deck(VALDATA / "t24_vacuum_regression.toml", out):
        REPORT["test24_vacuum_regression"] = {"pass": False, "error": "simulation failed"}
        return False

    data = read_hits_bin(out)
    if len(data) == 0:
        print("   no hits")
        REPORT["test24_vacuum_regression"] = {"pass": False, "error": "no hits"}
        return False

    ke_arr  = data[:, 2].astype(np.float64)
    mean_ke = float(np.mean(ke_arr))
    std_ke  = float(np.std(ke_arr))
    rel_std = std_ke / mean_ke if mean_ke > 0 else float("inf")
    tol = 1e-4

    ok = rel_std < tol
    print(f"   hits = {len(data)},  mean KE = {mean_ke:.4f} MeV,  "
          f"std/mean = {rel_std:.2e}  (tol {tol:.0e})")
    if not ok:
        print(f"   FAIL: std/mean {rel_std:.2e} ≥ {tol:.0e}  "
              "(Bethe-Bloch path active even without [density] block)")

    REPORT["test24_vacuum_regression"] = {
        "pass": ok,
        "n_hits": len(data),
        "mean_ke_MeV": round(mean_ke, 6),
        "rel_std": float(f"{rel_std:.3e}"),
        "tolerance": tol,
        "note": "TOML-path analog of test4; no [density] → exact B-only energy conservation.",
    }
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--build" in sys.argv or not BIN.exists():
        build()

    if not BIN.exists():
        sys.exit(f"Binary not found: {BIN}\nRun:  python3 validate.py --build")

    tests = [
        test1_regression,
        test2_zero_fields,
        test3_uniform_E,
        test4_B_energy_conservation,
        test5_pencil_tilted,
        test6_point_full_cone,
        test7_disk_spatial_spread,
        test8_energy_spread,
        test9_blur_conservation,
        test10_poisson_reproducibility,
        test11_exponential_spectrum,
        test12_relativistic_60mev,
        test13_tilted_geometry,
        test14_superimposed_fields,
        test15_adaptive_dt,
        test16_bethe_bloch,
        test17_analytic_hit,
        test18_larmor_radius,
        test19_exb_velocity_selector,
        test20_hits_bin_rebinning,
        test21_geometry_invariance,
        test22_field_compositing_linearity,
        test23_density_scaling,
        test24_vacuum_regression,
    ]

    results = {}
    for fn in tests:
        print()
        try:
            ok = fn()
        except Exception as ex:
            print(f"   exception: {ex}")
            ok = False
        results[fn.__name__] = ok
        print(f"   → {PASS if ok else FAIL}")

    print()
    print("─" * 52)
    all_pass = all(v for v in results.values() if v is not None)
    for name, ok in results.items():
        tag = PASS if ok else ("SKIP" if ok is None else FAIL)
        print(f"  {name:<35}  {tag}")
    print("─" * 52)

    # Write machine-readable report
    import datetime

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    report_path = ROOT / "output" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    full_report = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "all_pass": all_pass,
        "results": REPORT,
    }
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, cls=_NumpyEncoder)
    print(f"\nValidation report written to: {report_path}")

    sys.exit(0 if all_pass else 1)
