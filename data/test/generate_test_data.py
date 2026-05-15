#!/usr/bin/env python3
"""Generate test field and config for proton tracer"""

import numpy as np
import struct
import json
import os

# Create output directory
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Grid dimensions
nx, ny, nz = 64, 64, 128

# Physical bounds (meters)
x_min, x_max = -0.05, 0.05
y_min, y_max = -0.05, 0.05
z_min, z_max = -0.1, 0.2

# Create coordinate grids
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
z = np.linspace(z_min, z_max, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Create a simple Z-pinch field: B_theta proportional to 1/r
# B_theta = mu_0 * I / (2 * pi * r) for r > wire_radius
wire_radius = 0.005  # 5mm
current = 1e6  # 1 MA
mu_0 = 4e-7 * np.pi

R = np.sqrt(X**2 + Y**2)
R_safe = np.maximum(R, wire_radius)

# B_theta in cylindrical coords -> Bx, By in Cartesian
B_theta = mu_0 * current / (2 * np.pi * R_safe)
# Inside wire: B grows linearly with r
inside = R < wire_radius
B_theta_inside = (mu_0 * current / (2 * np.pi * wire_radius**2)) * R
B_theta[inside] = B_theta_inside[inside]

# Convert to Cartesian
theta = np.arctan2(Y, X)
Bx = -B_theta * np.sin(theta)
By = B_theta * np.cos(theta)
Bz = np.zeros_like(Bx)

# Write binary field file
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zpinch.bfld")
with open(output_path, 'wb') as f:
    # Header (64 bytes)
    f.write(b'BFLD')  # Magic
    f.write(struct.pack('<I', 1))  # Version
    f.write(struct.pack('<III', nx, ny, nz))  # Dimensions
    f.write(struct.pack('<6f', x_min, x_max, y_min, y_max, z_min, z_max))  # Bounds
    # Padding to 64 bytes
    f.write(b'\x00' * (64 - 4 - 4 - 12 - 24))

    # Data: nx * ny * nz * 3 floats (x, y, z order)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                f.write(struct.pack('<3f', Bx[ix, iy, iz], By[ix, iy, iz], Bz[ix, iy, iz]))

print(f"Wrote field to {output_path}")
print(f"  Grid: {nx}x{ny}x{nz}")
print(f"  Bounds: [{x_min}, {x_max}] x [{y_min}, {y_max}] x [{z_min}, {z_max}]")
print(f"  Max |B|: {np.max(np.sqrt(Bx**2 + By**2 + Bz**2)):.2f} T")

# Write config file
config = {
    "field_path": "zpinch.bfld",
    "source": {
        "source_type": "parallel",
        "n_protons": 100000,
        "energy_MeV": 14.7,
        "beam_center": [0.0, 0.0, -0.09],
        "beam_direction": [0.0, 0.0, 1.0],
        "beam_radius": 0.04,
        "angular_spread": 0.0,
        "point_position": None,
        "point_target": None,
        "detector_distance": 0.15,
        "detector_normal": [0.0, 0.0, 1.0]
    },
    "dt": 1e-12,
    "max_steps": 10000,
    "detector_bins": [512, 512]
}

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zpinch_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Wrote config to {config_path}")
