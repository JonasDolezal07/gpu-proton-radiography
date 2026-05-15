"""
Proton source configurations.

Defines how protons are emitted:
- Point source (like X-pinch)
- Parallel beam (uniform illumination)
- Custom distributions
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import struct
import json


class SourceType(Enum):
    POINT = "point"       # Diverging from single point
    PARALLEL = "parallel" # Uniform parallel beam
    CUSTOM = "custom"     # User-defined positions/velocities


@dataclass
class ProtonSource:
    """Configuration for proton emission."""
    source_type: SourceType

    # Common parameters
    n_protons: int
    energy_MeV: float

    # Point source parameters
    point_position: Optional[Tuple[float, float, float]] = None
    point_target: Optional[Tuple[float, float, float]] = None  # Aim toward this point
    angular_spread: float = 0.1  # Half-angle in radians

    # Parallel beam parameters
    beam_center: Optional[Tuple[float, float, float]] = None
    beam_direction: Optional[Tuple[float, float, float]] = None
    beam_radius: float = 0.01  # meters

    # Detector plane
    detector_distance: float = 0.15  # Distance from source/origin
    detector_normal: Tuple[float, float, float] = (0, 0, 1)  # Normal vector

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial positions and velocities.

        Returns:
            positions: (N, 3) float32 array
            velocities: (N, 3) float32 array
        """
        # Calculate speed (relativistic if needed)
        proton_rest_mass_MeV = 938.3
        c = 3e8

        if self.energy_MeV < 10:
            # Non-relativistic
            q = 1.602e-19
            m = 1.673e-27
            speed = np.sqrt(2 * self.energy_MeV * 1e6 * q / m)
        else:
            # Relativistic
            gamma = 1 + self.energy_MeV / proton_rest_mass_MeV
            beta = np.sqrt(1 - 1/gamma**2)
            speed = beta * c

        if self.source_type == SourceType.POINT:
            return self._generate_point_source(speed)
        elif self.source_type == SourceType.PARALLEL:
            return self._generate_parallel_beam(speed)
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")

    def _generate_point_source(self, speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate diverging beam from point."""
        n = self.n_protons
        src = np.array(self.point_position, dtype=np.float32)
        tgt = np.array(self.point_target or (0, 0, 0), dtype=np.float32)

        # Central direction
        central_dir = tgt - src
        central_dir = central_dir / np.linalg.norm(central_dir)

        # Build perpendicular basis
        if abs(central_dir[0]) < 0.9:
            perp1 = np.cross(central_dir, np.array([1, 0, 0]))
        else:
            perp1 = np.cross(central_dir, np.array([0, 1, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(central_dir, perp1)

        # Random angles
        theta = np.random.uniform(0, 2 * np.pi, n)
        phi = np.random.uniform(0, self.angular_spread, n)

        # Velocity directions
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        directions = (cos_phi[:, None] * central_dir +
                      sin_phi[:, None] * np.cos(theta)[:, None] * perp1 +
                      sin_phi[:, None] * np.sin(theta)[:, None] * perp2)

        positions = np.tile(src, (n, 1)).astype(np.float32)
        velocities = (directions * speed).astype(np.float32)

        return positions, velocities

    def _generate_parallel_beam(self, speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate uniform parallel beam."""
        n = self.n_protons
        center = np.array(self.beam_center or (0, 0, -0.1), dtype=np.float32)
        direction = np.array(self.beam_direction or (0, 0, 1), dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        # Build perpendicular basis
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        # Uniform disk distribution
        theta = np.random.uniform(0, 2 * np.pi, n)
        r = self.beam_radius * np.sqrt(np.random.uniform(0, 1, n))

        # Positions
        positions = (center +
                     r[:, None] * np.cos(theta)[:, None] * perp1 +
                     r[:, None] * np.sin(theta)[:, None] * perp2).astype(np.float32)

        # Velocities (all same direction)
        velocities = np.tile(direction * speed, (n, 1)).astype(np.float32)

        return positions, velocities

    def get_detector_plane(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get detector plane parameters.

        Returns:
            center: (3,) point on detector plane
            normal: (3,) normal vector
        """
        normal = np.array(self.detector_normal, dtype=np.float32)
        normal = normal / np.linalg.norm(normal)

        if self.source_type == SourceType.POINT:
            src = np.array(self.point_position)
            center = src + normal * self.detector_distance
        else:
            center = np.array(self.beam_center or (0, 0, 0)) + normal * self.detector_distance

        return center.astype(np.float32), normal

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "source_type": self.source_type.value,
            "n_protons": self.n_protons,
            "energy_MeV": self.energy_MeV,
            "point_position": self.point_position,
            "point_target": self.point_target,
            "angular_spread": self.angular_spread,
            "beam_center": self.beam_center,
            "beam_direction": self.beam_direction,
            "beam_radius": self.beam_radius,
            "detector_distance": self.detector_distance,
            "detector_normal": self.detector_normal
        }

    def export_binary(self, filepath: str, positions: np.ndarray, velocities: np.ndarray):
        """
        Export particle data for GPU.

        Format:
        - Header (32 bytes):
          - magic: 4 bytes "PRTC"
          - version: u32
          - n_particles: u32
          - reserved: padding to 32 bytes
        - Positions: n * 3 * float32
        - Velocities: n * 3 * float32
        """
        n = len(positions)
        with open(filepath, 'wb') as f:
            f.write(b'PRTC')
            f.write(struct.pack('II', 1, n))
            f.write(b'\x00' * (32 - 12))
            f.write(positions.astype(np.float32).tobytes())
            f.write(velocities.astype(np.float32).tobytes())

        # Metadata sidecar
        meta_path = filepath.replace('.bin', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class SimulationConfig:
    """Complete configuration for a proton radiography simulation."""
    field_path: str           # Path to field binary file
    source: ProtonSource
    dt: float = 1e-12         # Time step [s]
    max_steps: int = 10000    # Maximum integration steps
    detector_bins: Tuple[int, int] = (512, 512)  # Detector resolution

    def export(self, filepath: str):
        """Export complete config for Rust backend."""
        config = {
            "field_path": self.field_path,
            "source": self.source.to_dict(),
            "dt": self.dt,
            "max_steps": self.max_steps,
            "detector_bins": list(self.detector_bins)
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("=== Source Config Tests ===\n")

    # Test 1: Point source
    print("1. Point source...")
    point_src = ProtonSource(
        source_type=SourceType.POINT,
        n_protons=10000,
        energy_MeV=15.0,
        point_position=(0, 0, -0.05),
        point_target=(0, 0, 0),
        angular_spread=0.1,
        detector_distance=0.15
    )
    pos, vel = point_src.generate()
    print(f"   Positions shape: {pos.shape}")
    print(f"   Velocity magnitude: {np.linalg.norm(vel[0]):.3e} m/s")
    print(f"   Angular spread check: {np.std(vel[:, :2] / vel[:, 2:3]):.4f} rad")

    # Test 2: Parallel beam
    print("\n2. Parallel beam...")
    parallel_src = ProtonSource(
        source_type=SourceType.PARALLEL,
        n_protons=10000,
        energy_MeV=15.0,
        beam_center=(0, 0, -0.1),
        beam_direction=(0, 0, 1),
        beam_radius=0.015,
        detector_distance=0.2
    )
    pos, vel = parallel_src.generate()
    print(f"   Positions shape: {pos.shape}")
    print(f"   Beam radius check: max_r = {np.max(np.sqrt(pos[:, 0]**2 + pos[:, 1]**2))*1000:.1f} mm")
    print(f"   All same direction: {np.allclose(vel[0], vel[100])}")

    # Test 3: Export
    print("\n3. Exporting...")
    pos, vel = point_src.generate()
    point_src.export_binary('/Users/Jonas/Desktop/Everything/Projects/plasma/proton_tracer/test_particles.bin', pos, vel)
    print("   Saved to test_particles.bin")

    # Test 4: Full config
    print("\n4. Full simulation config...")
    config = SimulationConfig(
        field_path="test_field.bin",
        source=point_src,
        dt=1e-12,
        max_steps=5000,
        detector_bins=(512, 512)
    )
    config.export('/Users/Jonas/Desktop/Everything/Projects/plasma/proton_tracer/test_config.json')
    print("   Saved to test_config.json")

    print("\n=== All tests passed ===")
