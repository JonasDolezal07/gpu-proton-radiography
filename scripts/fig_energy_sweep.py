#!/usr/bin/env python3
"""
Figure: Energy sweep — synthetic radiographs at 6 proton energies.

Same z-pinch field and geometry; energies 3, 5, 10, 14.7, 30, 60 MeV.
Shows transition from non-relativistic (γ ≈ 1.003 at 3 MeV) to
mildly relativistic (γ ≈ 1.064 at 60 MeV) and the 1/p scaling of deflection.

Output: paper/figures/energy_sweep.pdf
"""

import sys, os, tempfile, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).parent.parent.resolve()
BIN  = ROOT / 'rust/target/release/proton_tracer'

ENERGIES = [3.0, 5.0, 10.0, 14.7, 30.0, 60.0]

FIELD_PATH = ROOT / 'data/instabilities/zpinch.bfld'
PIXELS     = 512
DET_HALF   = 250   # mm half-width

N_PARTICLES_FULL = 300_000
N_PARTICLES_FAST =  50_000

DECK_TEMPLATE = """
[field]
path = "{field_path}"

[source]
type               = "parallel"
energy_MeV         = {energy_MeV}
n_particles        = {n_particles}
beam_radius_mm     = 40.0
source_distance_mm = 80.0

[detector]
center_mm = [100.0, 0.0, 0.0]
width_mm  = 500.0
height_mm = 500.0
pixels    = [{pixels}, {pixels}]

[numerics]
dt_ps      = {dt_ps}
max_steps  = {max_steps}
"""

def dt_and_steps(energy_MeV):
    """Choose dt and max_steps appropriate for the proton energy."""
    # Faster protons need fewer steps to cross the field but smaller dt
    # for accuracy at high energy. Use 0.2 ps as default; scale down for
    # high-energy tail; add margin for deflected paths.
    if energy_MeV <= 5:
        return 1.0, 30000   # slow protons need longer time steps but more
    elif energy_MeV <= 15:
        return 0.2, 25000
    elif energy_MeV <= 35:
        return 0.1, 25000
    else:
        return 0.05, 30000  # 60 MeV — fast, needs small dt


def gamma_beta(energy_MeV):
    M_PC2 = 938.272  # MeV
    g = 1.0 + energy_MeV / M_PC2
    b = np.sqrt(1.0 - 1.0 / g**2)
    return g, b


def run_energy(energy_MeV, tmpdir, idx, n_particles=N_PARTICLES_FULL):
    dt, max_steps = dt_and_steps(energy_MeV)
    deck = tmpdir / f'deck_{idx}.toml'
    out  = tmpdir / f'out_{idx}'
    deck.write_text(DECK_TEMPLATE.format(
        field_path=FIELD_PATH, energy_MeV=energy_MeV,
        n_particles=n_particles, pixels=PIXELS, dt_ps=dt, max_steps=max_steps,
    ))
    env = os.environ.copy()
    env.setdefault('VK_ICD_FILENAMES', '/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json')
    env['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + env.get('DYLD_LIBRARY_PATH', '')
    r = subprocess.run(
        [str(BIN), 'run', str(deck), '-o', str(out), '--overwrite'],
        env=env, capture_output=True, text=True, cwd=ROOT,
    )
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError(f'prad failed at {energy_MeV} MeV')
    counts = np.frombuffer((out / 'counts' / 'raw_counts.bin').read_bytes(), dtype='<u4')
    n = int(len(counts)**0.5)
    return counts.reshape(n, n).astype(float)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true',
                    help=f'Use {N_PARTICLES_FAST:,} particles instead of {N_PARTICLES_FULL:,}')
    args = ap.parse_args()
    n_particles = N_PARTICLES_FAST if args.fast else N_PARTICLES_FULL

    imgs = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i, E in enumerate(ENERGIES):
            print(f'Running E = {E} MeV  ({i+1}/{len(ENERGIES)})…')
            imgs.append(run_energy(E, tmpdir, i, n_particles=n_particles))

    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8), constrained_layout=True)

    for ax, img, E in zip(axes.flat, imgs, ENERGIES):
        g, b = gamma_beta(E)
        vmax = np.percentile(img, 99.5)
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=max(vmax, 1))
        im = ax.imshow(img, origin='lower', norm=norm, cmap='inferno',
                       extent=[-DET_HALF, DET_HALF, -DET_HALF, DET_HALF])
        ax.set_title(f'{E} MeV\n(γ={g:.3f}, β={b:.3f})', fontsize=7)
        ax.set_xlabel('y (mm)', fontsize=6)
        ax.tick_params(labelsize=6)

    for ax in axes[:, 0]:
        ax.set_ylabel('z (mm)', fontsize=6)

    for i, ax in enumerate(axes.flat):
        ax.text(0.02, 0.96, f'({chr(ord("a")+i)})', transform=ax.transAxes,
                fontsize=7, color='white', va='top', fontweight='bold')

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.95)
    cbar.set_label('counts (sqrt scale)', fontsize=7)
    fig.suptitle('Energy sweep: z-pinch field', fontsize=9)

    out = ROOT / 'paper/figures/energy_sweep.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
