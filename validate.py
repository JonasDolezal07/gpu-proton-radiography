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
      2. mean(ke_MeV) within 20 % of T (correct distribution shape)
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
    mean_ok   = abs(mean_ke - T_MEV) / T_MEV < 0.20   # within 20 % of T
    spread_ok = (std_ke / mean_ke) > 0.3               # not monoenergetic

    ok = cutoff_ok and mean_ok and spread_ok
    print(f"   hits={len(ke_vals)}  mean={mean_ke:.4f} MeV  std={std_ke:.4f} MeV  max={max_ke:.4f} MeV")
    print(f"   (T={T_MEV} MeV, cutoff={CUTOFF} MeV)")
    if not cutoff_ok:
        print(f"   FAIL: max_ke {max_ke:.4f} > cutoff {CUTOFF} + 0.05 MeV")
    if not mean_ok:
        print(f"   FAIL: mean {mean_ke:.4f} not within 20% of T={T_MEV}")
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
    report_path = ROOT / "output" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    full_report = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "all_pass": all_pass,
        "results": REPORT,
    }
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nValidation report written to: {report_path}")

    sys.exit(0 if all_pass else 1)
