#!/usr/bin/env python3
"""
Figure: Full-orbit vs paraxial approximation — kink instability field.

Three field amplitudes (5%, 20%, 50% of kink_strong).
Top row: paraxial (straight-line path-integral deflection map).
Bottom row: full-orbit prad (relativistic Boris).

Output: paper/figures/paraxial_breakdown.pdf
"""

import sys, os, struct, tempfile, subprocess, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
BIN  = ROOT / 'rust/target/release/proton_tracer'

# ── physical constants ──────────────────────────────────────────────────────
M_P = 1.6726219e-27   # kg
C   = 2.99792458e8    # m/s
Q   = 1.60217663e-19  # C


def relativistic_rigidity(energy_MeV):
    """Magnetic rigidity p₀/q in T·m for a proton of given kinetic energy."""
    KE    = energy_MeV * 1e6 * Q
    gamma = 1.0 + KE / (M_P * C**2)
    beta  = np.sqrt(1.0 - 1.0 / gamma**2)
    p0    = gamma * M_P * beta * C
    return p0 / Q


def read_bfld(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'BFLD'
        version = struct.unpack('<I', f.read(4))[0]
        nx, ny, nz = struct.unpack('<III', f.read(12))
        bounds = struct.unpack('<6f', f.read(24))
        f.read(20)  # padding
        B = np.frombuffer(f.read(nx * ny * nz * 3 * 4), dtype='<f4').reshape(nx, ny, nz, 3).copy()
    return B, np.array(bounds), version


def write_bfld(path, B, bounds):
    nx, ny, nz = B.shape[:3]
    with open(path, 'wb') as f:
        f.write(b'BFLD')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<III', nx, ny, nz))
        f.write(struct.pack('<6f', *bounds))
        f.write(b'\x00' * 20)
        f.write(B.astype('<f4').tobytes())


def paraxial_radiograph(B_field, bounds, scale,
                         beam_radius_mm=40.0, n1d=300,
                         source_dist_mm=80.0, detector_dist_mm=100.0,
                         energy_MeV=14.7, det_half_mm=250.0, pixels=512):
    """
    Compute a paraxial radiograph using the lever-arm path-integral formula:

        δy = -(q/p₀) ∫ Bz(x, y₀, z₀) · (x_det - x) dx
        δz =  (q/p₀) ∫ By(x, y₀, z₀) · (x_det - x) dx

    Returns a (pixels × pixels) float array (counts per pixel).
    """
    B = B_field * scale
    rigidity = relativistic_rigidity(energy_MeV)  # T·m

    xmin, xmax = bounds[0], bounds[1]
    ymin, ymax = bounds[2], bounds[3]
    zmin, zmax = bounds[4], bounds[5]
    nx, ny, nz = B.shape[:3]

    x_det = detector_dist_mm * 1e-3   # m
    r_src = beam_radius_mm * 1e-3     # m

    # Source particle grid
    ys = np.linspace(-r_src, r_src, n1d)
    zs = np.linspace(-r_src, r_src, n1d)
    Y0, Z0 = np.meshgrid(ys, zs, indexing='ij')   # (n1d, n1d)
    in_beam = (Y0**2 + Z0**2) <= r_src**2

    # x sampling along the field (one per field voxel)
    x_vals = np.linspace(xmin, xmax, nx)
    dx = (xmax - xmin) / (nx - 1)
    lever = (x_det - x_vals)  # (nx,) — leverage of deflection at each x

    # Fractional indices into By/Bz arrays for all (y0, z0) at once
    iy_frac = (Y0 - ymin) / (ymax - ymin) * (ny - 1)  # (n1d, n1d)
    iz_frac = (Z0 - zmin) / (zmax - zmin) * (nz - 1)

    iy0 = np.clip(iy_frac.astype(int), 0, ny - 2)
    iz0 = np.clip(iz_frac.astype(int), 0, nz - 2)
    fy  = iy_frac - iy0
    fz  = iz_frac - iz0

    # For each (iy0, iz0), extract the B column along x — vectorised over grid
    # B shape: (nx, ny, nz, 3); extract By and Bz
    By = B[:, :, :, 1]  # (nx, ny, nz)
    Bz = B[:, :, :, 2]

    def bilin(F):
        """Bilinearly interpolate F[nx, ny, nz] on the (iy0,iy0+1) x (iz0,iz0+1) quad.
        Returns (nx, n1d, n1d)."""
        # F[:, iy0, iz0]  etc. using advanced indexing
        iy1 = iy0 + 1
        iz1 = iz0 + 1
        # weights shaped for broadcast: (1, n1d, n1d)
        w00 = ((1 - fy) * (1 - fz))[np.newaxis]
        w10 = (fy * (1 - fz))[np.newaxis]
        w01 = ((1 - fy) * fz)[np.newaxis]
        w11 = (fy * fz)[np.newaxis]
        return (w00 * F[:, iy0, iz0] +
                w10 * F[:, iy1, iz0] +
                w01 * F[:, iy0, iz1] +
                w11 * F[:, iy1, iz1])  # (nx, n1d, n1d)

    By_path = bilin(By)  # (nx, n1d, n1d)
    Bz_path = bilin(Bz)

    # Lever-arm integral: sum over x axis
    # lever shape: (nx,) → (nx, 1, 1)
    lev = lever[:, np.newaxis, np.newaxis]
    delta_z =  np.sum(By_path * lev, axis=0) * dx / rigidity   # (n1d, n1d)
    delta_y = -np.sum(Bz_path * lev, axis=0) * dx / rigidity

    y_det = (Y0 + delta_y)[in_beam]
    z_det = (Z0 + delta_z)[in_beam]

    # Convert to mm and histogram
    y_det_mm = y_det * 1e3
    z_det_mm = z_det * 1e3

    edges = np.linspace(-det_half_mm, det_half_mm, pixels + 1)
    H, _, _ = np.histogram2d(y_det_mm, z_det_mm, bins=[edges, edges])
    return H.T  # match imshow orientation (rows = z, cols = y)


def run_prad(bfld_path, scale, tmpdir, run_idx, n_particles=500_000,
             beam_radius_mm=40.0, energy_MeV=14.7,
             source_dist_mm=80.0, detector_dist_mm=100.0, pixels=512):
    """Run prad on a scaled version of bfld_path. Return (pixels×pixels) raw_counts."""
    B, bounds, _ = read_bfld(bfld_path)
    B_scaled = B * scale
    scaled_path = tmpdir / f'scaled_{run_idx}.bfld'
    write_bfld(scaled_path, B_scaled, bounds)

    deck_path = tmpdir / f'deck_{run_idx}.toml'
    out_dir   = tmpdir / f'out_{run_idx}'
    deck_path.write_text(f"""
[field]
path = "{scaled_path}"

[source]
type               = "parallel"
energy_MeV         = {energy_MeV}
n_particles        = {n_particles}
beam_radius_mm     = {beam_radius_mm}
source_distance_mm = {source_dist_mm}

[detector]
center_mm  = [{detector_dist_mm}, 0.0, 0.0]
width_mm   = 500.0
height_mm  = 500.0
pixels     = [{pixels}, {pixels}]

[numerics]
dt_ps      = 0.2
max_steps  = 25000
""")

    env = os.environ.copy()
    env.setdefault('VK_ICD_FILENAMES', '/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json')
    env['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + env.get('DYLD_LIBRARY_PATH', '')

    result = subprocess.run(
        [str(BIN), 'run', str(deck_path), '-o', str(out_dir), '--overwrite'],
        env=env, capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(f'prad failed for run {run_idx}')

    counts_file = out_dir / 'counts' / 'raw_counts.bin'
    raw = np.frombuffer(counts_file.read_bytes(), dtype='<u4')
    n = int(len(raw)**0.5)
    return raw.reshape(n, n).astype(float)


def percentile_norm(img, lo=1, hi=99.5):
    vmin = np.percentile(img[img > 0], lo) if img.any() else 0
    vmax = np.percentile(img, hi)
    return vmin, vmax


def main():
    bfld = ROOT / 'data/instabilities/kink_strong.bfld'
    scales = [0.05, 0.20, 0.50]
    labels = ['5%  (θ_max ≈ 0.2 rad)', '20% (θ_max ≈ 0.9 rad)', '50% (θ_max ≈ 2.2 rad)']

    print('Computing paraxial radiographs…')
    B0, bounds, _ = read_bfld(bfld)
    paraxial_imgs = [paraxial_radiograph(B0, bounds, s) for s in scales]

    print('Running full-orbit prad…')
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fullorbit_imgs = [run_prad(bfld, s, tmpdir, i) for i, s in enumerate(scales)]

    print('Plotting…')
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})

    row_labels = ['Paraxial', 'Full-orbit (prad)']
    col_labels = labels

    for col, (par, full, lbl) in enumerate(zip(paraxial_imgs, fullorbit_imgs, col_labels)):
        for row, (img, rlbl) in enumerate([(par, row_labels[0]), (full, row_labels[1])]):
            ax = axes[row, col]
            vmin, vmax = percentile_norm(full, lo=1, hi=99.5)
            if row == 0:
                # paraxial — use same colour scale as full-orbit for comparability
                norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
            else:
                norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
            im = ax.imshow(img, origin='lower', norm=norm, cmap='inferno',
                           extent=[-250, 250, -250, 250])
            if row == 0:
                ax.set_title(lbl, fontsize=7, pad=3)
            if col == 0:
                ax.set_ylabel(rlbl, fontsize=7)
            ax.set_xlabel('y (mm)', fontsize=6)
            if col == 0:
                ax.set_ylabel(f'{rlbl}\nz (mm)', fontsize=6)
            ax.tick_params(labelsize=6)

        # Quantitative label: peak deflection ratio
        par_sum  = paraxial_imgs[col].sum()
        full_max = fullorbit_imgs[col].max()
        par_max  = paraxial_imgs[col].max()
        if full_max > 0 and par_max > 0:
            ratio = par_max / full_max
            axes[0, col].text(0.97, 0.03, f'par/full peak = {ratio:.2f}',
                              transform=axes[0, col].transAxes,
                              ha='right', va='bottom', fontsize=5.5,
                              color='white')

    # Colourbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('counts (sqrt scale)', fontsize=7)

    # Panel labels
    for i, ax in enumerate(axes.flat):
        lbl = chr(ord('a') + i)
        ax.text(0.02, 0.96, f'({lbl})', transform=ax.transAxes,
                fontsize=7, color='white', va='top', fontweight='bold')

    fig.suptitle('Paraxial vs full-orbit: kink instability field', fontsize=9, y=1.01)
    fig.tight_layout()

    out = ROOT / 'paper/figures/paraxial_breakdown.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')

    # Print quantitative comparison numbers for the paper text
    rigidity = relativistic_rigidity(14.7)
    B0_arr, bounds0, _ = read_bfld(bfld)
    dx = (bounds0[1]-bounds0[0]) / B0_arr.shape[0]
    int_By_max = np.abs(np.sum(B0_arr[:, :, :, 1], axis=0) * dx).max()
    for s in scales:
        theta_max = s * int_By_max / rigidity
        print(f"scale={s:.2f}: max |∫By dx| = {s*int_By_max:.3f} T·m → θ_max = {theta_max:.2f} rad = {np.degrees(theta_max):.1f}°")


if __name__ == '__main__':
    main()
