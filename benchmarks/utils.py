"""Shared utilities for the benchmark suite."""

import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parent.parent
BIN       = ROOT / "rust" / "target" / "release" / "proton_tracer"
BENCH_DIR = Path(__file__).parent

RESULTS_DIR = BENCH_DIR / "results"
PLOTS_DIR   = BENCH_DIR / "plots"
FIELDS_DIR  = BENCH_DIR / "fields"

# Physical constants (SI)
PROTON_Q = 1.602176634e-19    # C
PROTON_M = 1.67262192369e-27  # kg


def setup_dirs():
    for d in [RESULTS_DIR, PLOTS_DIR, FIELDS_DIR, RESULTS_DIR / "runs"]:
        d.mkdir(parents=True, exist_ok=True)


def vulkan_env():
    """Return os.environ copy with macOS MoltenVK vars set if needed."""
    env = os.environ.copy()
    icd     = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    brew    = Path("/opt/homebrew/lib")
    if icd.exists() and "VK_ICD_FILENAMES" not in env:
        env["VK_ICD_FILENAMES"] = str(icd)
    if brew.exists():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        if str(brew) not in existing:
            env["DYLD_LIBRARY_PATH"] = f"{brew}:{existing}".rstrip(":")
    return env


def proton_speed(energy_MeV):
    """Non-relativistic proton speed (m/s) for given kinetic energy."""
    KE_J = energy_MeV * PROTON_Q * 1e6
    return float(np.sqrt(2.0 * KE_J / PROTON_M))


def proton_momentum(energy_MeV):
    """Non-relativistic proton momentum (kg·m/s)."""
    return PROTON_M * proton_speed(energy_MeV)


def run_tracer(deck_path, out_dir, overrides=None, overwrite=True, verbose=False):
    """
    Run `proton_tracer run <deck> -o <out_dir> [--set k=v ...]`.
    Returns parsed metadata.json dict. Raises RuntimeError on non-zero exit.
    """
    cmd = [str(BIN), "run", str(deck_path), "-o", str(out_dir)]
    if overwrite:
        cmd.append("--overwrite")
    for k, v in (overrides or {}).items():
        cmd += ["--set", f"{k}={v}"]

    env = vulkan_env()
    result = subprocess.run(
        cmd, env=env, cwd=ROOT,
        capture_output=not verbose, text=True,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-600:]
        raise RuntimeError(f"proton_tracer failed (exit {result.returncode}):\n{tail}")

    with open(Path(out_dir) / "metadata.json") as f:
        return json.load(f)


def read_hits_bin(run_dir):
    """
    Read hits.bin from a run directory.
    Returns (y_mm, z_mm, energy_MeV) as (N,3) float32 array, or empty array.
    Format: u32 count | (y_mm f32, z_mm f32, energy_MeV f32) × count
    """
    path = Path(run_dir) / "counts" / "hits.bin"
    data = path.read_bytes()
    n = np.frombuffer(data[:4], dtype="<u4")[0]
    if n == 0:
        return np.zeros((0, 3), dtype="<f4")
    return np.frombuffer(data[4:], dtype="<f4").reshape(n, 3).copy()


def read_raw_counts(run_dir):
    """
    Read raw_counts.bin from a run directory. Returns (H, W) uint32 array.
    The GPU renderer always writes at DETECTOR_RESOLUTION = 1024, regardless
    of the deck's [detector] pixels setting.
    """
    path = Path(run_dir) / "counts" / "raw_counts.bin"
    data = np.fromfile(path, dtype="<u4")
    side = int(np.sqrt(len(data)))   # always 1024
    return data.reshape(side, side)


def read_bfld(path):
    """
    Read a .bfld file.
    Returns (header: dict, B: ndarray (nx,ny,nz,3) float32, E or None).
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"BFLD":
            raise ValueError(f"Bad BFLD magic: {magic!r}")
        version, = struct.unpack("<I", f.read(4))
        nx, ny, nz = struct.unpack("<III", f.read(12))
        bounds = struct.unpack("<6f", f.read(24))
        f.read(20)   # padding to 64-byte header
        n = nx * ny * nz * 3
        B = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(nx, ny, nz, 3).copy()
        E = None
        if version >= 2:
            E = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(nx, ny, nz, 3).copy()

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    header = dict(version=version, nx=nx, ny=ny, nz=nz,
                  x_min=float(x_min), x_max=float(x_max),
                  y_min=float(y_min), y_max=float(y_max),
                  z_min=float(z_min), z_max=float(z_max))
    return header, B, E


def write_bfld(path, B, bounds, E=None):
    """
    Write a .bfld file. B (and optionally E) must be (nx,ny,nz,3) float32.
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max) in metres.
    """
    nx, ny, nz = B.shape[:3]
    version = 2 if E is not None else 1
    with open(path, "wb") as f:
        f.write(b"BFLD")
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<III", nx, ny, nz))
        f.write(struct.pack("<6f", *bounds))
        f.write(b"\x00" * 20)
        f.write(B.astype("<f4").tobytes())
        if E is not None:
            f.write(E.astype("<f4").tobytes())


def paraxial_radiograph(bfld_path, scale_B,
                        beam_radius_mm, detector_center_mm,
                        detector_width_mm, detector_height_mm,
                        detector_pixels, energy_MeV,
                        n_rays=300_000, seed=42):
    """
    Compute a paraxial-approximation radiograph for a parallel beam.

    Integrates B_perp along the unperturbed straight-line path through the
    field for each ray and projects predicted hit positions onto the detector.

    Paraxial deflection angles (beam in +x, small deflection limit):
        theta_y = -(q/p) * integral(Bz dx)
        theta_z =  (q/p) * integral(By dx)

    Returns:
        counts   : (H, W) float32 array of predicted hit density
        theta_y  : (n_rays,) deflection angles in y [rad]
        theta_z  : (n_rays,) deflection angles in z [rad]
    """
    from scipy.interpolate import RegularGridInterpolator

    rng = np.random.default_rng(seed)

    header, B, _ = read_bfld(bfld_path)
    B = B * float(scale_B)

    nx, ny, nz = header["nx"], header["ny"], header["nz"]
    x_min, x_max = header["x_min"], header["x_max"]
    y_min, y_max = header["y_min"], header["y_max"]
    z_min, z_max = header["z_min"], header["z_max"]

    p = proton_momentum(energy_MeV)

    # Source: uniform disk of radius beam_radius_mm in y-z plane
    beam_r_m = beam_radius_mm * 1e-3
    n_over = int(n_rays * 1.4)
    ys = rng.uniform(-beam_r_m, beam_r_m, n_over)
    zs = rng.uniform(-beam_r_m, beam_r_m, n_over)
    inside = (ys**2 + zs**2) <= beam_r_m**2
    ys, zs = ys[inside][:n_rays], zs[inside][:n_rays]
    n = len(ys)

    # Scipy interpolators for By and Bz (fill with 0 outside field)
    xs_g = np.linspace(x_min, x_max, nx)
    ys_g = np.linspace(y_min, y_max, ny)
    zs_g = np.linspace(z_min, z_max, nz)

    By_interp = RegularGridInterpolator(
        (xs_g, ys_g, zs_g), B[:, :, :, 1],
        method="linear", bounds_error=False, fill_value=0.0)
    Bz_interp = RegularGridInterpolator(
        (xs_g, ys_g, zs_g), B[:, :, :, 2],
        method="linear", bounds_error=False, fill_value=0.0)

    # Integrate B along x at fixed (y0, z0) for all rays simultaneously
    n_x = 256
    x_pts = np.linspace(x_min, x_max, n_x)
    dx = (x_max - x_min) / n_x

    # Query shape: (n * n_x, 3)
    x_rep = np.tile(x_pts, n)
    y_rep = np.repeat(ys, n_x)
    z_rep = np.repeat(zs, n_x)
    pts = np.column_stack([x_rep, y_rep, z_rep])

    By_v = By_interp(pts).reshape(n, n_x)
    Bz_v = Bz_interp(pts).reshape(n, n_x)

    # F_y = q(v×B)_y = -q*v*Bz,  so theta_y = -(q/p)*int(Bz dx)
    # F_z = q(v×B)_z =  q*v*By,  so theta_z =  (q/p)*int(By dx)
    theta_y = -(PROTON_Q / p) * Bz_v.sum(axis=1) * dx
    theta_z =  (PROTON_Q / p) * By_v.sum(axis=1) * dx

    # Project to detector (lever arm = detector_x - field_x_max)
    x_det  = detector_center_mm[0] * 1e-3
    lever  = x_det - x_max

    hit_y_mm = (ys + theta_y * lever) * 1e3
    hit_z_mm = (zs + theta_z * lever) * 1e3

    # Bin into detector pixel grid
    W, H = detector_pixels
    half_w = detector_width_mm / 2.0
    half_h = detector_height_mm / 2.0

    col = ((hit_y_mm + half_w) / detector_width_mm  * W).astype(int)
    row = ((hit_z_mm + half_h) / detector_height_mm * H).astype(int)

    on_det = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    counts = np.zeros((H, W), dtype=np.float32)
    np.add.at(counts, (row[on_det], col[on_det]), 1.0)

    return counts, theta_y, theta_z
