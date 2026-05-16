#!/usr/bin/env python3
"""Generate test field and config for proton tracer

Z-pinch geometry:
- Pinch (current) flows along Z-axis
- Azimuthal B-field wraps around Z
- Protons travel in +X direction (perpendicular to pinch)
- Detector catches protons on +X side
"""

import numpy as np
import struct
import json
import os

# Create output directory
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Grid dimensions
nx, ny, nz = 64, 64, 128

# Physical bounds (meters)
# The pinch runs along Z, so we want:
# - X: protons travel through this dimension
# - Y: transverse to proton beam
# - Z: along the pinch (long dimension)
pinch_radius = 0.02  # 2cm pinch radius
pinch_length = 0.15  # 15cm long pinch

# Grid extends slightly beyond pinch for proton paths
x_min, x_max = -0.05, 0.05   # 10cm total, protons cross this
y_min, y_max = -0.05, 0.05   # 10cm transverse
z_min, z_max = -0.075, 0.075 # 15cm along pinch

# Create coordinate grids
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
z = np.linspace(z_min, z_max, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Z-pinch field: B_theta = mu_0 * I / (2 * pi * r)
# Only inside the pinch radius - outside is vacuum (B=0)
wire_radius = 0.002  # 2mm wire core
current = 500e3  # 500 kA
mu_0 = 4e-7 * np.pi

# Radial distance from Z-axis
R = np.sqrt(X**2 + Y**2)

# B_theta calculation
B_theta = np.zeros_like(R)

# Inside wire core: B grows linearly with r
inside_wire = R < wire_radius
B_theta[inside_wire] = (mu_0 * current / (2 * np.pi * wire_radius**2)) * R[inside_wire]

# Between wire and pinch radius: B = mu_0 * I / (2*pi*r)
in_pinch = (R >= wire_radius) & (R < pinch_radius)
B_theta[in_pinch] = mu_0 * current / (2 * np.pi * R[in_pinch])

# Outside pinch radius: B = 0 (this makes it a "pipe")
# Already zero from initialization

# Convert to Cartesian (B_theta -> Bx, By)
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
print(f"  Pinch radius: {pinch_radius*100:.1f} cm")
print(f"  Current: {current/1e3:.0f} kA")
print(f"  Max |B|: {np.max(np.sqrt(Bx**2 + By**2 + Bz**2)):.2f} T")

# Write config file
# Protons travel in +X direction, perpendicular to pinch
# Start at x = -0.08 (outside field), end at detector at x = +0.10
config = {
    "field_path": "zpinch.bfld",
    "source": {
        "source_type": "parallel",
        "n_protons": 1000000,  # 1 million particles
        "energy_MeV": 14.7,
        "beam_center": [-0.08, 0.0, 0.0],  # Start outside field, centered on pinch
        "beam_direction": [1.0, 0.0, 0.0],  # Travel in +X
        "beam_radius": 0.04,  # 4cm radius beam covers the pinch
        "angular_spread": 0.0,
        "point_position": None,
        "point_target": None,
        "detector_distance": 0.05,  # 5cm beyond field exit
        "detector_normal": [1.0, 0.0, 0.0]  # Facing -X (toward source)
    },
    "dt": 1e-12,
    "max_steps": 10000,
    "detector_bins": [512, 512]
}

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zpinch_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Wrote config to {config_path}")
print(f"\nGeometry:")
print(f"  Protons: start at x={config['source']['beam_center'][0]}, travel in +X")
print(f"  Pinch: cylinder along Z-axis, radius={pinch_radius*100:.0f}cm")
print(f"  Detector: at x={x_max + config['source']['detector_distance']:.2f}m")
