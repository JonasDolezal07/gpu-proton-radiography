"""Build a TOML deck string from Python kwargs."""

from __future__ import annotations

from typing import Optional, Tuple


def _vec(v: Tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{x}" for x in v) + "]"


def build_deck(
    field_path: str,
    *,
    source: str = "parallel",
    energy_MeV: float = 14.7,
    n_particles: int = 100_000,
    beam_radius_mm: float = 40.0,
    source_distance_mm: float = 80.0,
    angular_spread_deg: float = 20.0,
    beam_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    detector_distance_mm: float = 100.0,
    detector_size_mm: Tuple[float, float] = (500.0, 500.0),
    detector_up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    dt_ps: float = 0.2,
    max_steps: int = 25_000,
    scale_B: float = 1.0,
    scale_E: float = 0.0,
    colormap: str = "rcf",
    exposure: float = 1.0,
    write_processed_counts: bool = False,
) -> str:
    """Return a TOML deck string ready to write to disk."""

    # Detector sits downstream along the beam axis.
    det_cx = beam_direction[0] * detector_distance_mm
    det_cy = beam_direction[1] * detector_distance_mm
    det_cz = beam_direction[2] * detector_distance_mm

    source_section = f"""[source]
type = "{source}"
direction = {_vec(beam_direction)}
beam_radius_mm = {beam_radius_mm}
source_distance_mm = {source_distance_mm}
angular_spread_deg = {angular_spread_deg}
energy_MeV = {energy_MeV}
n_particles = {n_particles}
"""

    deck = f"""[field]
path = "{field_path}"
scale_B = {scale_B}
scale_E = {scale_E}

{source_section}
[detector]
center_mm = {_vec((det_cx, det_cy, det_cz))}
normal = {_vec(beam_direction)}
up = {_vec(detector_up)}
width_mm = {detector_size_mm[0]}
height_mm = {detector_size_mm[1]}
pixels = [1024, 1024]

[numerics]
dt_ps = {dt_ps}
max_steps = {max_steps}

[render]
scale = "log"
colormap = "{colormap}"
exposure = {exposure}

[output]
write_raw_counts = true
write_processed_counts = {"true" if write_processed_counts else "false"}
write_png = true
write_metadata = true
"""
    return deck
