"""
Standardized field format for proton radiography.

This module defines the interface for arbitrary electromagnetic fields
that can be fed into the tracer. Supports:
- Grid-based fields (from MHD simulations)
- Analytical fields (for testing/prototyping)
- Composite fields (superposition)

Output format is designed for GPU consumption:
- Field data as contiguous float32 arrays
- Metadata as simple dict for serialization
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import json
import struct


@dataclass
class FieldBounds:
    """Spatial bounds of the field domain."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.x_min, self.x_max,
            self.y_min, self.y_max,
            self.z_min, self.z_max
        ], dtype=np.float32)

    @property
    def size(self) -> Tuple[float, float, float]:
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min
        )

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2
        )


@dataclass
class FieldMetadata:
    """Metadata about the field."""
    name: str
    description: str = ""
    units_length: str = "m"      # meters, R_E, mm, etc.
    units_field: str = "T"       # Tesla
    source: str = "unknown"      # "analytical", "gorgon", "flash", etc.
    timestamp: Optional[float] = None  # simulation time if applicable
    extra: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "units_length": self.units_length,
            "units_field": self.units_field,
            "source": self.source,
            "timestamp": self.timestamp,
            "extra": self.extra or {}
        }


class Field(ABC):
    """Abstract base class for electromagnetic fields."""

    @abstractmethod
    def sample(self, positions: np.ndarray) -> np.ndarray:
        """
        Sample field at given positions.

        Args:
            positions: (N, 3) array of positions

        Returns:
            (N, 3) array of B-field vectors
        """
        pass

    @abstractmethod
    def get_bounds(self) -> FieldBounds:
        """Return the spatial bounds of the field."""
        pass

    @abstractmethod
    def get_metadata(self) -> FieldMetadata:
        """Return metadata about the field."""
        pass

    def to_grid(self, resolution: Tuple[int, int, int]) -> 'GridField':
        """
        Convert any field to a grid representation.
        Useful for GPU upload.
        """
        bounds = self.get_bounds()
        nx, ny, nz = resolution

        # Create sample points
        x = np.linspace(bounds.x_min, bounds.x_max, nx)
        y = np.linspace(bounds.y_min, bounds.y_max, ny)
        z = np.linspace(bounds.z_min, bounds.z_max, nz)

        # Sample on grid
        data = np.zeros((nx, ny, nz, 3), dtype=np.float32)

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                positions = np.column_stack([
                    np.full(nz, xi),
                    np.full(nz, yj),
                    z
                ])
                data[i, j, :, :] = self.sample(positions)

        meta = self.get_metadata()
        meta.description += " [converted to grid]"

        return GridField(data, bounds, meta)


class GridField(Field):
    """
    Field defined on a regular 3D grid.
    This is the format used for GPU upload.
    """

    def __init__(self,
                 data: np.ndarray,
                 bounds: FieldBounds,
                 metadata: FieldMetadata):
        """
        Args:
            data: (nx, ny, nz, 3) array of B-field vectors
            bounds: Spatial bounds
            metadata: Field metadata
        """
        assert data.ndim == 4 and data.shape[3] == 3
        self.data = data.astype(np.float32)
        self.bounds = bounds
        self.metadata = metadata

        self.nx, self.ny, self.nz = data.shape[:3]
        self.dx = (bounds.x_max - bounds.x_min) / (self.nx - 1)
        self.dy = (bounds.y_max - bounds.y_min) / (self.ny - 1)
        self.dz = (bounds.z_max - bounds.z_min) / (self.nz - 1)

    def sample(self, positions: np.ndarray) -> np.ndarray:
        """Trilinear interpolation sampling."""
        # Normalize to grid coordinates
        gx = (positions[:, 0] - self.bounds.x_min) / self.dx
        gy = (positions[:, 1] - self.bounds.y_min) / self.dy
        gz = (positions[:, 2] - self.bounds.z_min) / self.dz

        # Integer indices (clamped)
        ix = np.clip(gx.astype(np.int32), 0, self.nx - 2)
        iy = np.clip(gy.astype(np.int32), 0, self.ny - 2)
        iz = np.clip(gz.astype(np.int32), 0, self.nz - 2)

        # Fractional parts
        fx = np.clip(gx - ix, 0, 1)
        fy = np.clip(gy - iy, 0, 1)
        fz = np.clip(gz - iz, 0, 1)

        # Trilinear interpolation
        w000 = (1-fx) * (1-fy) * (1-fz)
        w001 = (1-fx) * (1-fy) * fz
        w010 = (1-fx) * fy * (1-fz)
        w011 = (1-fx) * fy * fz
        w100 = fx * (1-fy) * (1-fz)
        w101 = fx * (1-fy) * fz
        w110 = fx * fy * (1-fz)
        w111 = fx * fy * fz

        result = (w000[:, None] * self.data[ix, iy, iz] +
                  w001[:, None] * self.data[ix, iy, iz+1] +
                  w010[:, None] * self.data[ix, iy+1, iz] +
                  w011[:, None] * self.data[ix, iy+1, iz+1] +
                  w100[:, None] * self.data[ix+1, iy, iz] +
                  w101[:, None] * self.data[ix+1, iy, iz+1] +
                  w110[:, None] * self.data[ix+1, iy+1, iz] +
                  w111[:, None] * self.data[ix+1, iy+1, iz+1])

        return result

    def get_bounds(self) -> FieldBounds:
        return self.bounds

    def get_metadata(self) -> FieldMetadata:
        return self.metadata

    def export_binary(self, filepath: str):
        """
        Export to binary format for Rust/GPU consumption.

        Format:
        - Header (64 bytes):
          - magic: 4 bytes "BFLD"
          - version: u32
          - nx, ny, nz: 3x u32
          - bounds: 6x f32
          - reserved: padding to 64 bytes
        - Data: nx*ny*nz*3 float32 values (contiguous)
        """
        with open(filepath, 'wb') as f:
            # Magic
            f.write(b'BFLD')
            # Version
            f.write(struct.pack('I', 1))
            # Grid dimensions
            f.write(struct.pack('III', self.nx, self.ny, self.nz))
            # Bounds
            f.write(struct.pack('6f',
                self.bounds.x_min, self.bounds.x_max,
                self.bounds.y_min, self.bounds.y_max,
                self.bounds.z_min, self.bounds.z_max))
            # Padding to 64 bytes
            f.write(b'\x00' * (64 - 4 - 4 - 12 - 24))
            # Data
            f.write(self.data.tobytes())

        # Also write metadata as JSON sidecar
        meta_path = filepath.replace('.bin', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

    @classmethod
    def from_binary(cls, filepath: str) -> 'GridField':
        """Load from binary format."""
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            assert magic == b'BFLD', f"Invalid magic: {magic}"

            version = struct.unpack('I', f.read(4))[0]
            assert version == 1

            nx, ny, nz = struct.unpack('III', f.read(12))
            bounds_arr = struct.unpack('6f', f.read(24))

            f.read(64 - 4 - 4 - 12 - 24)  # skip padding

            data = np.frombuffer(f.read(), dtype=np.float32)
            data = data.reshape((nx, ny, nz, 3))

        bounds = FieldBounds(*bounds_arr)

        # Try to load metadata
        meta_path = filepath.replace('.bin', '_meta.json')
        try:
            with open(meta_path, 'r') as f:
                meta_dict = json.load(f)
            metadata = FieldMetadata(**meta_dict)
        except:
            metadata = FieldMetadata(name="imported", source="binary")

        return cls(data, bounds, metadata)


class AnalyticalField(Field):
    """Field defined by an analytical function."""

    def __init__(self,
                 func: Callable[[np.ndarray], np.ndarray],
                 bounds: FieldBounds,
                 metadata: FieldMetadata):
        """
        Args:
            func: Function (N,3) -> (N,3) mapping positions to B-field
            bounds: Spatial bounds (for reference)
            metadata: Field metadata
        """
        self.func = func
        self.bounds = bounds
        self.metadata = metadata

    def sample(self, positions: np.ndarray) -> np.ndarray:
        return self.func(positions)

    def get_bounds(self) -> FieldBounds:
        return self.bounds

    def get_metadata(self) -> FieldMetadata:
        return self.metadata


class CompositeField(Field):
    """Superposition of multiple fields."""

    def __init__(self, fields: list, metadata: Optional[FieldMetadata] = None):
        self.fields = fields
        self._metadata = metadata or FieldMetadata(
            name="composite",
            description=f"Superposition of {len(fields)} fields",
            source="composite"
        )

    def sample(self, positions: np.ndarray) -> np.ndarray:
        result = np.zeros((len(positions), 3), dtype=np.float32)
        for field in self.fields:
            result += field.sample(positions)
        return result

    def get_bounds(self) -> FieldBounds:
        # Union of all bounds
        all_bounds = [f.get_bounds() for f in self.fields]
        return FieldBounds(
            x_min=min(b.x_min for b in all_bounds),
            x_max=max(b.x_max for b in all_bounds),
            y_min=min(b.y_min for b in all_bounds),
            y_max=max(b.y_max for b in all_bounds),
            z_min=min(b.z_min for b in all_bounds),
            z_max=max(b.z_max for b in all_bounds),
        )

    def get_metadata(self) -> FieldMetadata:
        return self._metadata


# ============================================================
# Built-in analytical field generators
# ============================================================

def zpinch_field(current: float,
                 radius: float,
                 z_min: float,
                 z_max: float,
                 sausage_amp: float = 0.0,
                 sausage_k: float = 0.0,
                 kink_amp: float = 0.0,
                 kink_k: float = 0.0) -> AnalyticalField:
    """
    Create a Z-pinch field with optional instabilities.

    Args:
        current: Total current [A]
        radius: Wire radius [m]
        z_min, z_max: Axial extent of the pinch
        sausage_amp: Relative sausage amplitude (0-1)
        sausage_k: Sausage wavenumber [rad/m]
        kink_amp: Kink displacement [m]
        kink_k: Kink wavenumber [rad/m]
    """
    mu0 = 4 * np.pi * 1e-7

    def field_func(positions: np.ndarray) -> np.ndarray:
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        B = np.zeros_like(positions)

        in_z = (z >= z_min) & (z <= z_max)

        # Perturbed radius (sausage)
        a_z = radius * (1 + sausage_amp * np.cos(sausage_k * z))

        # Axis displacement (kink)
        axis_x = kink_amp * np.cos(kink_k * z)
        axis_y = kink_amp * np.sin(kink_k * z)

        # Distance from perturbed axis
        dx = x - axis_x
        dy = y - axis_y
        r = np.sqrt(dx**2 + dy**2)
        r_safe = np.maximum(r, 1e-10)

        # B-field magnitude
        B_theta = np.zeros_like(r)
        outside = in_z & (r >= a_z)
        inside = in_z & (r < a_z)

        B_theta[outside] = mu0 * current / (2 * np.pi * r_safe[outside])
        B_theta[inside] = mu0 * current * r_safe[inside] / (2 * np.pi * a_z[inside]**2)

        # Convert to Cartesian
        theta = np.arctan2(dy, dx)
        B[:, 0] = -B_theta * np.sin(theta)
        B[:, 1] = B_theta * np.cos(theta)

        return B.astype(np.float32)

    # Bounds: extend a bit beyond the pinch
    extent = max(radius * 5, 0.02)
    bounds = FieldBounds(
        x_min=-extent, x_max=extent,
        y_min=-extent, y_max=extent,
        z_min=z_min - 0.01, z_max=z_max + 0.01
    )

    metadata = FieldMetadata(
        name="Z-pinch",
        description=f"I={current/1e3:.0f}kA, r={radius*1e3:.1f}mm",
        source="analytical"
    )
    if sausage_amp > 0:
        metadata.description += f", sausage={sausage_amp:.1%}"
    if kink_amp > 0:
        metadata.description += f", kink={kink_amp*1e3:.1f}mm"

    return AnalyticalField(field_func, bounds, metadata)


def uniform_field(B: Tuple[float, float, float],
                  bounds: FieldBounds) -> AnalyticalField:
    """Uniform magnetic field."""
    B_vec = np.array(B, dtype=np.float32)

    def field_func(positions: np.ndarray) -> np.ndarray:
        return np.tile(B_vec, (len(positions), 1))

    metadata = FieldMetadata(
        name="Uniform",
        description=f"B=({B[0]:.2e}, {B[1]:.2e}, {B[2]:.2e}) T",
        source="analytical"
    )

    return AnalyticalField(field_func, bounds, metadata)


def dipole_field(moment: float,
                 center: Tuple[float, float, float] = (0, 0, 0),
                 bounds: Optional[FieldBounds] = None) -> AnalyticalField:
    """
    Magnetic dipole field (like Earth's).

    Args:
        moment: Magnetic dipole moment [A·m²]
        center: Position of dipole
        bounds: Field bounds (defaults to large region)
    """
    mu0 = 4 * np.pi * 1e-7
    c = np.array(center, dtype=np.float32)

    def field_func(positions: np.ndarray) -> np.ndarray:
        r_vec = positions - c
        r = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r = np.maximum(r, 1e-10)  # avoid singularity
        r_hat = r_vec / r

        # Dipole along z-axis
        m_hat = np.array([0, 0, 1], dtype=np.float32)
        m_dot_r = np.sum(r_hat * m_hat, axis=1, keepdims=True)

        # B = (mu0 / 4pi) * (3(m·r)r - m) / r³
        prefactor = mu0 * moment / (4 * np.pi * r**3)
        B = prefactor * (3 * m_dot_r * r_hat - m_hat)

        return B.astype(np.float32)

    if bounds is None:
        extent = 1e7  # 10,000 km default
        bounds = FieldBounds(-extent, extent, -extent, extent, -extent, extent)

    metadata = FieldMetadata(
        name="Dipole",
        description=f"moment={moment:.2e} A·m²",
        source="analytical"
    )

    return AnalyticalField(field_func, bounds, metadata)


# ============================================================
# Loaders for common simulation formats
# ============================================================

def load_gorgon_hdf5(filepath: str,
                     bounds: Optional[FieldBounds] = None) -> GridField:
    """
    Load field from GORGON HDF5 output.

    Expects dataset 'Bvec_c' with shape (nx, ny, nz, 3).
    """
    import h5py

    with h5py.File(filepath, 'r') as f:
        data = f['Bvec_c'][:].astype(np.float32)

    nx, ny, nz = data.shape[:3]

    if bounds is None:
        # Default bounds (user should specify proper ones)
        bounds = FieldBounds(-1, 1, -1, 1, -1, 1)

    metadata = FieldMetadata(
        name="GORGON",
        description=f"Grid {nx}×{ny}×{nz}",
        source="gorgon"
    )

    return GridField(data, bounds, metadata)


def load_vtk(filepath: str) -> GridField:
    """Load field from VTK structured grid."""
    # TODO: implement VTK loading
    raise NotImplementedError("VTK loading not yet implemented")


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("=== Field Format Tests ===\n")

    # Test 1: Z-pinch analytical
    print("1. Creating Z-pinch field...")
    zpinch = zpinch_field(
        current=100e3,
        radius=0.003,
        z_min=-0.02,
        z_max=0.02,
        sausage_amp=0.3,
        sausage_k=2*np.pi/0.015
    )
    print(f"   {zpinch.get_metadata().name}: {zpinch.get_metadata().description}")

    # Sample some points
    test_points = np.array([
        [0.005, 0.0, 0.0],
        [0.0, 0.005, 0.0],
        [0.001, 0.001, 0.01]
    ])
    B = zpinch.sample(test_points)
    print(f"   Sample B-field magnitudes: {np.linalg.norm(B, axis=1)}")

    # Test 2: Convert to grid
    print("\n2. Converting to grid (64³)...")
    grid = zpinch.to_grid((64, 64, 64))
    print(f"   Grid shape: {grid.data.shape}")
    print(f"   Data size: {grid.data.nbytes / 1e6:.1f} MB")

    # Test 3: Export to binary
    print("\n3. Exporting to binary...")
    grid.export_binary('/Users/Jonas/Desktop/Everything/Projects/plasma/proton_tracer/test_field.bin')
    print("   Saved to test_field.bin")

    # Test 4: Reload
    print("\n4. Reloading from binary...")
    grid2 = GridField.from_binary('/Users/Jonas/Desktop/Everything/Projects/plasma/proton_tracer/test_field.bin')
    print(f"   Loaded grid shape: {grid2.data.shape}")
    print(f"   Data matches: {np.allclose(grid.data, grid2.data)}")

    # Test 5: Composite field
    print("\n5. Creating composite field...")
    uniform = uniform_field((0, 0, 0.1), zpinch.get_bounds())
    composite = CompositeField([zpinch, uniform])
    B_composite = composite.sample(test_points)
    print(f"   Composite B magnitudes: {np.linalg.norm(B_composite, axis=1)}")

    print("\n=== All tests passed ===")
