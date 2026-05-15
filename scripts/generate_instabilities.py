#!/usr/bin/env python3
"""
Generate magnetic field files for various plasma instabilities.

Instability types:
- Z-pinch: Baseline azimuthal field (B_theta ~ 1/r)
- Sausage (m=0): Axisymmetric pinching - field varies along z
- Kink (m=1): Helical displacement - field has helical perturbation

Output format: .bfld binary files compatible with proton_tracer
"""

import numpy as np
import struct
from pathlib import Path


def write_bfld(filename: str, field: np.ndarray, bounds: tuple):
    """
    Write field data in .bfld format.

    Format (64-byte header):
    - Magic: "BFLD" (4 bytes)
    - Version: u32 = 1 (4 bytes)
    - nx, ny, nz: 3x u32 (12 bytes)
    - bounds: 6x f32 (x_min, x_max, y_min, y_max, z_min, z_max) (24 bytes)
    - padding: 20 bytes to reach 64 bytes
    - Data: nx * ny * nz * 3 * f32 (Bx, By, Bz)
    """
    nz, ny, nx, _ = field.shape
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    with open(filename, 'wb') as f:
        # Header (64 bytes total)
        header = bytearray(64)
        header[0:4] = b'BFLD'
        struct.pack_into('<I', header, 4, 1)  # version
        struct.pack_into('<III', header, 8, nx, ny, nz)
        struct.pack_into('<ffffff', header, 20, x_min, x_max, y_min, y_max, z_min, z_max)
        # Bytes 44-63 are padding (already zeros)
        f.write(header)

        # Data (flatten in C order: z, y, x, component)
        field_flat = field.astype(np.float32).flatten()
        f.write(field_flat.tobytes())

    print(f"Wrote {filename}: {nx}x{ny}x{nz}, bounds={bounds}")


def generate_zpinch(nx: int, ny: int, nz: int, bounds: tuple,
                    I0: float = 1e6, a: float = 0.005,
                    pinch_radius: float = None, pinch_length: float = None) -> np.ndarray:
    """
    Generate baseline z-pinch field in a CYLINDRICAL region.

    The Z-pinch is a cylinder along the z-axis with:
    - Current flowing in +z direction
    - Azimuthal B-field circling the z-axis
    - Field falls off smoothly outside the pinch radius

    B_theta = mu0 * I / (2*pi*r) for r > a (outside current channel)
    B_theta = mu0 * I * r / (2*pi*a^2) for r < a (inside current channel)

    Field is zero outside the pinch cylinder.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Default pinch dimensions: cylinder that fits in the grid
    if pinch_radius is None:
        pinch_radius = min(x_max, y_max) * 0.8  # 80% of grid extent
    if pinch_length is None:
        pinch_length = (z_max - z_min) * 0.9  # 90% of z-extent

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Cylindrical coordinates (r from z-axis)
    r = np.sqrt(X**2 + Y**2)
    r = np.maximum(r, 1e-10)  # Avoid division by zero

    # B_theta magnitude
    mu0 = 4 * np.pi * 1e-7
    B_theta = np.where(
        r > a,
        mu0 * I0 / (2 * np.pi * r),
        mu0 * I0 * r / (2 * np.pi * a**2)
    )

    # Apply cylindrical cutoff: field only exists inside the pinch cylinder
    # Smooth falloff at the edges using a tanh profile
    z_center = (z_max + z_min) / 2
    half_length = pinch_length / 2

    # Radial falloff: smooth transition at pinch_radius
    radial_falloff = 0.5 * (1 - np.tanh((r - pinch_radius) / (pinch_radius * 0.1)))

    # Axial falloff: smooth transition at pinch ends
    axial_falloff = 0.5 * (1 - np.tanh((np.abs(Z - z_center) - half_length) / (half_length * 0.1)))

    # Combined falloff
    B_theta = B_theta * radial_falloff * axial_falloff

    # Convert to Cartesian: B_x = -B_theta * sin(theta), B_y = B_theta * cos(theta)
    cos_theta = X / r
    sin_theta = Y / r

    Bx = -B_theta * sin_theta
    By = B_theta * cos_theta
    Bz = np.zeros_like(Bx)

    # Stack into (nz, ny, nx, 3) array
    field = np.stack([Bx, By, Bz], axis=-1)
    field = np.transpose(field, (2, 1, 0, 3))  # Reorder to (nz, ny, nx, 3)

    return field


def generate_sausage(nx: int, ny: int, nz: int, bounds: tuple,
                     I0: float = 1e6, a: float = 0.005,
                     perturbation: float = 0.3, wavelength: float = 0.02,
                     pinch_radius: float = None, pinch_length: float = None) -> np.ndarray:
    """
    Generate sausage instability (m=0 mode) in a CYLINDRICAL region.

    The current channel radius varies sinusoidally along z:
    a(z) = a0 * (1 + epsilon * sin(k*z))

    This creates regions where the field is stronger (pinched) and weaker (bulged).
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    if pinch_radius is None:
        pinch_radius = min(x_max, y_max) * 0.8
    if pinch_length is None:
        pinch_length = (z_max - z_min) * 0.9

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Cylindrical coordinates
    r = np.sqrt(X**2 + Y**2)
    r = np.maximum(r, 1e-10)

    # Perturbed radius: a(z) = a0 * (1 + epsilon * sin(k*z))
    k = 2 * np.pi / wavelength
    a_z = a * (1 + perturbation * np.sin(k * Z))

    # B_theta magnitude with z-dependent radius
    mu0 = 4 * np.pi * 1e-7
    B_theta = np.where(
        r > a_z,
        mu0 * I0 / (2 * np.pi * r),
        mu0 * I0 * r / (2 * np.pi * a_z**2)
    )

    # Cylindrical cutoff with smooth falloff
    z_center = (z_max + z_min) / 2
    half_length = pinch_length / 2
    radial_falloff = 0.5 * (1 - np.tanh((r - pinch_radius) / (pinch_radius * 0.1)))
    axial_falloff = 0.5 * (1 - np.tanh((np.abs(Z - z_center) - half_length) / (half_length * 0.1)))

    # The sausage instability also creates a small B_r component
    B_r = -perturbation * k * a * np.cos(k * Z) * B_theta * 0.1

    # Apply falloff
    B_theta = B_theta * radial_falloff * axial_falloff
    B_r = B_r * radial_falloff * axial_falloff

    # Convert to Cartesian
    cos_theta = X / r
    sin_theta = Y / r

    Bx = B_r * cos_theta - B_theta * sin_theta
    By = B_r * sin_theta + B_theta * cos_theta
    Bz = np.zeros_like(Bx)

    field = np.stack([Bx, By, Bz], axis=-1)
    field = np.transpose(field, (2, 1, 0, 3))

    return field


def generate_kink(nx: int, ny: int, nz: int, bounds: tuple,
                  I0: float = 1e6, a: float = 0.005,
                  perturbation: float = 0.2, wavelength: float = 0.03,
                  pinch_radius: float = None, pinch_length: float = None) -> np.ndarray:
    """
    Generate kink instability (m=1 mode) in a CYLINDRICAL region.

    The current channel is displaced helically:
    x_offset = epsilon * a * cos(k*z)
    y_offset = epsilon * a * sin(k*z)

    This creates a corkscrewing field structure.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    if pinch_radius is None:
        pinch_radius = min(x_max, y_max) * 0.8
    if pinch_length is None:
        pinch_length = (z_max - z_min) * 0.9

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Helical displacement of the current channel axis
    k = 2 * np.pi / wavelength
    x_offset = perturbation * a * np.cos(k * Z)
    y_offset = perturbation * a * np.sin(k * Z)

    # Shifted coordinates (distance from the displaced axis)
    X_shifted = X - x_offset
    Y_shifted = Y - y_offset
    r = np.sqrt(X_shifted**2 + Y_shifted**2)
    r = np.maximum(r, 1e-10)

    # Also compute r from original axis for falloff
    r_orig = np.sqrt(X**2 + Y**2)
    r_orig = np.maximum(r_orig, 1e-10)

    # B_theta magnitude (around the displaced axis)
    mu0 = 4 * np.pi * 1e-7
    B_theta = np.where(
        r > a,
        mu0 * I0 / (2 * np.pi * r),
        mu0 * I0 * r / (2 * np.pi * a**2)
    )

    # Cylindrical cutoff with smooth falloff (using original axis)
    z_center = (z_max + z_min) / 2
    half_length = pinch_length / 2
    radial_falloff = 0.5 * (1 - np.tanh((r_orig - pinch_radius) / (pinch_radius * 0.1)))
    axial_falloff = 0.5 * (1 - np.tanh((np.abs(Z - z_center) - half_length) / (half_length * 0.1)))
    falloff = radial_falloff * axial_falloff

    # Convert to Cartesian (relative to displaced axis)
    cos_theta = X_shifted / r
    sin_theta = Y_shifted / r

    Bx = -B_theta * sin_theta * falloff
    By = B_theta * cos_theta * falloff

    # The kink also induces a B_z component from the helical geometry
    Bz = -perturbation * k * a * B_theta * 0.1 * falloff

    field = np.stack([Bx, By, Bz], axis=-1)
    field = np.transpose(field, (2, 1, 0, 3))

    return field


def create_config(output_dir: Path, name: str, field_file: str,
                  n_protons: int = 1_000_000, energy_MeV: float = 14.7):
    """Create JSON config file for the simulation."""
    config = f'''{{
  "field_path": "{field_file}",
  "source": {{
    "source_type": "parallel",
    "n_protons": {n_protons},
    "energy_MeV": {energy_MeV},
    "beam_center": [-0.08, 0.0, 0.0],
    "beam_direction": [1.0, 0.0, 0.0],
    "beam_radius": 0.04,
    "angular_spread": 0.0,
    "point_position": null,
    "point_target": null,
    "detector_distance": 0.05,
    "detector_normal": [1.0, 0.0, 0.0]
  }},
  "dt": 1e-12,
  "max_steps": 10000,
  "detector_bins": [512, 512]
}}
'''
    config_path = output_dir / f"{name}.json"
    with open(config_path, 'w') as f:
        f.write(config)
    print(f"Created config: {config_path}")


def main():
    # Output directory
    output_dir = Path(__file__).parent.parent / "data" / "instabilities"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grid parameters - CYLINDRICAL pinch geometry
    # The pinch is elongated along z (current flow direction)
    # x,y are transverse (radial) directions
    nx, ny, nz = 64, 64, 128

    # Physical dimensions for realistic Z-pinch:
    # - Pinch length: ~6cm along z-axis
    # - Radial extent: ~4cm radius
    # Grid bounds slightly larger to contain the field smoothly
    bounds = (-0.05, 0.05, -0.05, 0.05, -0.04, 0.04)  # 10cm x 10cm x 8cm

    # Pinch parameters
    pinch_radius = 0.035   # 3.5cm - field falls off beyond this
    pinch_length = 0.06    # 6cm - axial extent of pinch

    print("Generating CYLINDRICAL Z-pinch fields...")
    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"Bounds: {bounds}")
    print(f"Pinch: radius={pinch_radius*100:.1f}cm, length={pinch_length*100:.1f}cm")
    print()

    # Generate baseline z-pinch
    print("1. Z-pinch (baseline cylindrical)...")
    field = generate_zpinch(nx, ny, nz, bounds,
                           pinch_radius=pinch_radius, pinch_length=pinch_length)
    write_bfld(str(output_dir / "zpinch.bfld"), field, bounds)
    create_config(output_dir, "zpinch", "zpinch.bfld")

    # Generate sausage instability (weak)
    print("\n2. Sausage instability (weak, 10%)...")
    field = generate_sausage(nx, ny, nz, bounds, perturbation=0.1, wavelength=0.015,
                            pinch_radius=pinch_radius, pinch_length=pinch_length)
    write_bfld(str(output_dir / "sausage_weak.bfld"), field, bounds)
    create_config(output_dir, "sausage_weak", "sausage_weak.bfld")

    # Generate sausage instability (strong)
    print("\n3. Sausage instability (strong, 30%)...")
    field = generate_sausage(nx, ny, nz, bounds, perturbation=0.3, wavelength=0.015,
                            pinch_radius=pinch_radius, pinch_length=pinch_length)
    write_bfld(str(output_dir / "sausage_strong.bfld"), field, bounds)
    create_config(output_dir, "sausage_strong", "sausage_strong.bfld")

    # Generate kink instability (weak)
    print("\n4. Kink instability (weak, 10%)...")
    field = generate_kink(nx, ny, nz, bounds, perturbation=0.1, wavelength=0.02,
                         pinch_radius=pinch_radius, pinch_length=pinch_length)
    write_bfld(str(output_dir / "kink_weak.bfld"), field, bounds)
    create_config(output_dir, "kink_weak", "kink_weak.bfld")

    # Generate kink instability (strong)
    print("\n5. Kink instability (strong, 25%)...")
    field = generate_kink(nx, ny, nz, bounds, perturbation=0.25, wavelength=0.02,
                         pinch_radius=pinch_radius, pinch_length=pinch_length)
    write_bfld(str(output_dir / "kink_strong.bfld"), field, bounds)
    create_config(output_dir, "kink_strong", "kink_strong.bfld")

    # Generate mixed instability (sausage + kink)
    print("\n6. Mixed instability (sausage + kink)...")
    field_sausage = generate_sausage(nx, ny, nz, bounds, perturbation=0.2, wavelength=0.015,
                                    pinch_radius=pinch_radius, pinch_length=pinch_length)
    field_kink = generate_kink(nx, ny, nz, bounds, perturbation=0.15, wavelength=0.025,
                              pinch_radius=pinch_radius, pinch_length=pinch_length)
    field_mixed = (field_sausage + field_kink) / 2
    write_bfld(str(output_dir / "mixed.bfld"), field_mixed, bounds)
    create_config(output_dir, "mixed", "mixed.bfld")

    print("\n" + "="*50)
    print("Generated CYLINDRICAL fields in:", output_dir)
    print("\nTo run simulations:")
    print(f"  ./target/release/proton_tracer {output_dir}/zpinch.json")
    print(f"  ./target/release/proton_tracer {output_dir}/sausage_strong.json")
    print(f"  ./target/release/proton_tracer {output_dir}/kink_strong.json")
    print("\nFor batch mode (no visualization):")
    print(f"  ./target/release/proton_tracer {output_dir}/sausage_strong.json --batch")


if __name__ == "__main__":
    main()
