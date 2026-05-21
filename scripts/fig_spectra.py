#!/usr/bin/env python3
"""
Figure: Three source energy spectra — same z-pinch field, same geometry.

Left:   monoenergetic 14.7 MeV
Centre: Gaussian 14.7 MeV ± 5 %
Right:  TNSA exponential T = 3 MeV, E_cut = 40 MeV

Output: paper/figures/spectra_comparison.pdf
"""

import sys, os, tempfile, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
BIN  = ROOT / 'rust/target/release/proton_tracer'

DECKS = [
    # (label, toml source block, max_steps, dt_ps)
    (
        'Monoenergetic\n14.7 MeV',
        '[source]\ntype               = "parallel"\nenergy_MeV         = 14.7\nn_particles        = 500000\nbeam_radius_mm     = 40.0\nsource_distance_mm = 80.0',
        25000, 0.2,
    ),
    (
        'Gaussian\n14.7 MeV ± 5 %',
        '[source]\ntype                  = "parallel"\nenergy_MeV            = 14.7\nenergy_spread_percent = 5.0\nn_particles           = 500000\nbeam_radius_mm        = 40.0\nsource_distance_mm    = 80.0',
        25000, 0.2,
    ),
    (
        'TNSA exponential\nT = 3 MeV, E_cut = 40 MeV',
        '[source]\ntype               = "parallel"\nenergy_MeV         = 14.7\ntemperature_MeV    = 3.0\ncutoff_mev         = 40.0\nn_particles        = 500000\nbeam_radius_mm     = 40.0\nsource_distance_mm = 80.0',
        50000, 0.1,
    ),
]

FIELD_BLOCK = f"""
[field]
path = "{ROOT / 'data/instabilities/zpinch.bfld'}"

[detector]
center_mm = [100.0, 0.0, 0.0]
width_mm  = 500.0
height_mm = 500.0
pixels    = [512, 512]
"""


def run_deck(deck_toml, out_dir):
    env = os.environ.copy()
    env.setdefault('VK_ICD_FILENAMES', '/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json')
    env['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + env.get('DYLD_LIBRARY_PATH', '')
    r = subprocess.run(
        [str(BIN), 'run', str(deck_toml), '-o', str(out_dir), '--overwrite'],
        env=env, capture_output=True, text=True, cwd=ROOT,
    )
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError('prad failed')
    counts = np.frombuffer((out_dir / 'counts' / 'raw_counts.bin').read_bytes(), dtype='<u4')
    n = int(len(counts)**0.5)
    return counts.reshape(n, n).astype(float)


def main():
    imgs = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i, (lbl, src_block, max_steps, dt) in enumerate(DECKS):
            print(f'Running spectrum {i+1}/3: {lbl.replace(chr(10), " ")}')
            deck = tmpdir / f'deck_{i}.toml'
            deck.write_text(
                FIELD_BLOCK +
                f'\n{src_block}\n\n[numerics]\ndt_ps = {dt}\nmax_steps = {max_steps}\n'
            )
            imgs.append(run_deck(deck, tmpdir / f'out_{i}'))

    # Shared normalisation across all three images
    vmax = max(np.percentile(img, 99.5) for img in imgs)

    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), constrained_layout=True)

    for ax, img, (lbl, *_) in zip(axes, imgs, DECKS):
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
        im = ax.imshow(img, origin='lower', norm=norm, cmap='inferno',
                       extent=[-250, 250, -250, 250])
        ax.set_title(lbl, fontsize=7.5)
        ax.set_xlabel('y (mm)', fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0].set_ylabel('z (mm)', fontsize=7)

    # Panel labels
    for i, ax in enumerate(axes):
        ax.text(0.02, 0.96, f'({chr(ord("a")+i)})', transform=ax.transAxes,
                fontsize=7, color='white', va='top', fontweight='bold')

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.95)
    cbar.set_label('counts (sqrt scale)', fontsize=7)

    out = ROOT / 'paper/figures/spectra_comparison.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
