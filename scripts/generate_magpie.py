#!/usr/bin/env python3
"""
Generate realistic MAGPIE-like magnetic field for proton radiography.

MAGPIE (Mega Ampere Generator for Plasma Implosion Experiments) at Imperial College:
- Peak current: ~1.4 MA
- Rise time: ~240 ns
- Wire array Z-pinches with typical radius 8-16 mm
- Proton radiography with D3He backlighter (3 MeV or 14.7 MeV protons)

This generates fields representative of:
1. Wire array implosion with ablation streams
2. Developed sausage/kink instabilities during stagnation
3. Multiple wire positions creating interference patterns
"""

import numpy as np
import struct
from pathlib import Path


def write_bfld(filename: str, field: np.ndarray, bounds: tuple):
    """Write field data in .bfld format."""
    nz, ny, nx, _ = field.shape
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    with open(filename, 'wb') as f:
        header = bytearray(64)
        header[0:4] = b'BFLD'
        struct.pack_into('<I', header, 4, 1)
        struct.pack_into('<III', header, 8, nx, ny, nz)
        struct.pack_into('<ffffff', header, 20, x_min, x_max, y_min, y_max, z_min, z_max)
        f.write(header)
        field_flat = field.astype(np.float32).flatten()
        f.write(field_flat.tobytes())

    print(f"Wrote {filename}: {nx}x{ny}x{nz}, bounds={bounds}")


def generate_magpie_wire_array(nx: int, ny: int, nz: int, bounds: tuple,
                                n_wires: int = 16,
                                wire_array_radius: float = 0.012,  # 12mm typical
                                I_total: float = 1.4e6,  # 1.4 MA
                                wire_radius: float = 0.0005,  # 0.5mm current channel
                                pinch_length: float = 0.02,  # 20mm anode-cathode gap
                                instability_amplitude: float = 0.15,
                                instability_wavelength: float = 0.003) -> np.ndarray:
    """
    Generate MAGPIE-style wire array field with instabilities.

    Creates superposition of current channels from multiple wires,
    plus perturbations representing ablation and instability development.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    mu0 = 4 * np.pi * 1e-7
    I_per_wire = I_total / n_wires

    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)

    # Wire positions around the array
    wire_angles = np.linspace(0, 2*np.pi, n_wires, endpoint=False)

    k = 2 * np.pi / instability_wavelength
    z_center = (z_max + z_min) / 2
    half_length = pinch_length / 2

    for i, angle in enumerate(wire_angles):
        # Wire nominal position
        wx = wire_array_radius * np.cos(angle)
        wy = wire_array_radius * np.sin(angle)

        # Add helical perturbation (kink-like) with phase offset per wire
        phase = angle  # Each wire has different instability phase
        wx_perturb = instability_amplitude * wire_radius * np.cos(k * Z + phase) * 20
        wy_perturb = instability_amplitude * wire_radius * np.sin(k * Z + phase) * 20

        # Distance from this wire
        dx = X - (wx + wx_perturb)
        dy = Y - (wy + wy_perturb)
        r_wire = np.sqrt(dx**2 + dy**2)
        r_wire = np.maximum(r_wire, 1e-10)

        # B_theta from this wire
        B_theta_wire = np.where(
            r_wire > wire_radius,
            mu0 * I_per_wire / (2 * np.pi * r_wire),
            mu0 * I_per_wire * r_wire / (2 * np.pi * wire_radius**2)
        )

        # Convert to Cartesian (circling this wire)
        cos_t = dx / r_wire
        sin_t = dy / r_wire

        Bx += -B_theta_wire * sin_t
        By += B_theta_wire * cos_t

    # Apply axial falloff at anode/cathode
    axial_falloff = 0.5 * (1 - np.tanh((np.abs(Z - z_center) - half_length) / (half_length * 0.15)))

    # Apply radial falloff beyond the array
    r_from_axis = np.sqrt(X**2 + Y**2)
    outer_radius = wire_array_radius * 2.5
    radial_falloff = 0.5 * (1 - np.tanh((r_from_axis - outer_radius) / (outer_radius * 0.2)))

    falloff = axial_falloff * radial_falloff

    Bx *= falloff
    By *= falloff
    Bz *= falloff

    field = np.stack([Bx, By, Bz], axis=-1)
    field = np.transpose(field, (2, 1, 0, 3))

    return field


def generate_magpie_stagnation(nx: int, ny: int, nz: int, bounds: tuple,
                                I0: float = 1.4e6,
                                stagnation_radius: float = 0.002,  # 2mm stagnated
                                pinch_length: float = 0.015,
                                sausage_amp: float = 0.25,
                                sausage_wavelength: float = 0.002,
                                kink_amp: float = 0.20,
                                kink_wavelength: float = 0.004) -> np.ndarray:
    """
    Generate field during stagnation phase with developed instabilities.

    After implosion, the plasma stagnates on axis with strong m=0 (sausage)
    and m=1 (kink) instabilities visible in proton radiographs.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    mu0 = 4 * np.pi * 1e-7

    z_center = (z_max + z_min) / 2
    half_length = pinch_length / 2

    # Sausage perturbation (radius varies with z)
    k_sausage = 2 * np.pi / sausage_wavelength
    a_z = stagnation_radius * (1 + sausage_amp * np.sin(k_sausage * Z))

    # Kink perturbation (axis displaces helically)
    k_kink = 2 * np.pi / kink_wavelength
    x_offset = kink_amp * stagnation_radius * np.cos(k_kink * Z) * 5
    y_offset = kink_amp * stagnation_radius * np.sin(k_kink * Z) * 5

    # Distance from displaced axis
    X_shifted = X - x_offset
    Y_shifted = Y - y_offset
    r = np.sqrt(X_shifted**2 + Y_shifted**2)
    r = np.maximum(r, 1e-10)

    # B_theta with sausage modulation
    B_theta = np.where(
        r > a_z,
        mu0 * I0 / (2 * np.pi * r),
        mu0 * I0 * r / (2 * np.pi * a_z**2)
    )

    # Additional radial component from sausage
    B_r = -sausage_amp * k_sausage * stagnation_radius * np.cos(k_sausage * Z) * B_theta * 0.15

    # Axial and radial falloff
    r_orig = np.sqrt(X**2 + Y**2)
    outer_radius = 0.025  # 25mm
    axial_falloff = 0.5 * (1 - np.tanh((np.abs(Z - z_center) - half_length) / (half_length * 0.12)))
    radial_falloff = 0.5 * (1 - np.tanh((r_orig - outer_radius) / (outer_radius * 0.15)))
    falloff = axial_falloff * radial_falloff

    # Convert to Cartesian
    cos_theta = X_shifted / r
    sin_theta = Y_shifted / r

    Bx = (B_r * cos_theta - B_theta * sin_theta) * falloff
    By = (B_r * sin_theta + B_theta * cos_theta) * falloff
    Bz = -kink_amp * k_kink * stagnation_radius * B_theta * 0.08 * falloff

    field = np.stack([Bx, By, Bz], axis=-1)
    field = np.transpose(field, (2, 1, 0, 3))

    return field


def create_config(output_dir: Path, name: str, field_file: str,
                  n_protons: int = 2_000_000,
                  energy_MeV: float = 14.7,
                  source_type: str = "point",
                  source_distance: float = 0.05,  # 50mm before field (field starts at x=-0.03)
                  detector_distance: float = 0.20,  # 200mm after field exit
                  detector_bins: int = 1024):
    """
    Create JSON config with MAGPIE-like geometry.

    MAGPIE proton radiography typically uses:
    - D3He backlighter (14.7 MeV protons, or 3 MeV D-D)
    - Point projection geometry with ~10-25x magnification
    - Source before field, detector after field

    Geometry (beam travels in +X direction):
      Source (x=-0.05) --> Field (x=-0.03 to +0.03) --> Detector (x=+0.03 + distance)

    Field bounds: x=[-0.03, 0.03], so:
      - source_distance=0.05 puts source at x=-0.05 (before field entry at x=-0.03)
      - detector_distance=0.20 puts detector at x=0.03+0.20=0.23 (after field exit)
    """
    # Source position: before the field (field x_min = -0.03)
    source_x = -source_distance  # e.g., -0.05

    if source_type == "point":
        config = f'''{{
  "field_path": "{field_file}",
  "source": {{
    "source_type": "point",
    "n_protons": {n_protons},
    "energy_MeV": {energy_MeV},
    "point_position": [{source_x}, 0.0, 0.0],
    "point_target": [0.0, 0.0, 0.0],
    "angular_spread": 0.05,
    "beam_center": null,
    "beam_direction": [1.0, 0.0, 0.0],
    "beam_radius": 0.03,
    "detector_distance": {detector_distance},
    "detector_normal": [1.0, 0.0, 0.0]
  }},
  "dt": 5e-13,
  "max_steps": 20000,
  "detector_bins": [{detector_bins}, {detector_bins}]
}}
'''
    else:
        config = f'''{{
  "field_path": "{field_file}",
  "source": {{
    "source_type": "parallel",
    "n_protons": {n_protons},
    "energy_MeV": {energy_MeV},
    "beam_center": [{source_x}, 0.0, 0.0],
    "beam_direction": [1.0, 0.0, 0.0],
    "beam_radius": 0.04,
    "angular_spread": 0.0,
    "point_position": null,
    "point_target": null,
    "detector_distance": {detector_distance},
    "detector_normal": [1.0, 0.0, 0.0]
  }},
  "dt": 5e-13,
  "max_steps": 20000,
  "detector_bins": [{detector_bins}, {detector_bins}]
}}
'''

    config_path = output_dir / f"{name}.json"
    with open(config_path, 'w') as f:
        f.write(config)
    print(f"Created config: {config_path}")


def main():
    output_dir = Path(__file__).parent.parent / "data" / "magpie"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Higher resolution for detailed caustics
    nx, ny, nz = 128, 128, 256

    # MAGPIE-scale bounds: ~6cm x 6cm x 4cm
    bounds = (-0.03, 0.03, -0.03, 0.03, -0.02, 0.02)

    print("="*60)
    print("Generating MAGPIE-realistic fields")
    print("="*60)
    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"Bounds: {bounds} (6cm x 6cm x 4cm)")
    print()

    # 1. Wire array during implosion (16-wire array)
    print("1. Wire array implosion (16 wires, 12mm radius)...")
    field = generate_magpie_wire_array(
        nx, ny, nz, bounds,
        n_wires=16,
        wire_array_radius=0.012,
        I_total=1.4e6,
        instability_amplitude=0.10,
        instability_wavelength=0.004
    )
    write_bfld(str(output_dir / "magpie_wires.bfld"), field, bounds)
    create_config(output_dir, "magpie_wires", "magpie_wires.bfld",
                  n_protons=2_000_000, energy_MeV=14.7, source_type="point")

    # 2. Stagnation with mild instabilities
    print("\n2. Stagnation (mild instabilities)...")
    field = generate_magpie_stagnation(
        nx, ny, nz, bounds,
        I0=1.4e6,
        stagnation_radius=0.002,
        sausage_amp=0.15,
        sausage_wavelength=0.002,
        kink_amp=0.10,
        kink_wavelength=0.005
    )
    write_bfld(str(output_dir / "magpie_stagnation_mild.bfld"), field, bounds)
    create_config(output_dir, "magpie_stagnation_mild", "magpie_stagnation_mild.bfld",
                  n_protons=2_000_000, energy_MeV=14.7, source_type="point")

    # 3. Stagnation with strong instabilities (most dramatic caustics)
    print("\n3. Stagnation (strong instabilities - dramatic caustics)...")
    field = generate_magpie_stagnation(
        nx, ny, nz, bounds,
        I0=1.4e6,
        stagnation_radius=0.0015,
        sausage_amp=0.30,
        sausage_wavelength=0.0015,
        kink_amp=0.25,
        kink_wavelength=0.004
    )
    write_bfld(str(output_dir / "magpie_stagnation_strong.bfld"), field, bounds)
    create_config(output_dir, "magpie_stagnation_strong", "magpie_stagnation_strong.bfld",
                  n_protons=2_000_000, energy_MeV=14.7, source_type="point")

    # 4. D-D protons (3 MeV) for comparison
    print("\n4. Stagnation with 3 MeV D-D protons...")
    create_config(output_dir, "magpie_stagnation_3MeV", "magpie_stagnation_strong.bfld",
                  n_protons=2_000_000, energy_MeV=3.0, source_type="point")

    # 5. Parallel beam version (for cleaner caustic visualization)
    print("\n5. Parallel beam through stagnation...")
    create_config(output_dir, "magpie_parallel", "magpie_stagnation_strong.bfld",
                  n_protons=2_000_000, energy_MeV=14.7, source_type="parallel")

    print("\n" + "="*60)
    print("Generated MAGPIE-realistic fields in:", output_dir)
    print("\nRecommended for realistic caustics:")
    print(f"  cd /Users/Jonas/Desktop/Everything/Projects/plasma/gpu_proton_tracer/rust")
    print(f"  cargo run --release -- ../data/magpie/magpie_stagnation_strong.json")
    print("\nAlternatives:")
    print(f"  - magpie_wires.json       : Wire array during implosion")
    print(f"  - magpie_stagnation_mild.json : Mild instabilities")
    print(f"  - magpie_stagnation_3MeV.json : Lower energy protons (more deflection)")
    print(f"  - magpie_parallel.json    : Parallel beam (cleaner caustics)")
    print("="*60)


if __name__ == "__main__":
    main()
