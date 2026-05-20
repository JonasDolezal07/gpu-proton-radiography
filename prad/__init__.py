"""
prad — GPU-accelerated proton radiography.

Quick start::

    import prad

    result = prad.run(
        field="path/to/field.bfld",
        energy_MeV=14.7,
        n_particles=100_000,
        source_distance_mm=80.0,
        detector_distance_mm=100.0,
    )
    result.show()
    counts = result.raw_counts   # numpy uint32 array, always 1024×1024
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

from ._field import GridField
from ._result import RunResult

Field = GridField  # convenient alias

__all__ = ["run", "Field", "GridField", "RunResult"]
__version__ = "0.3.0"


def run(
    field: Union[str, Path, GridField],
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
    energy_spread_percent: float = 0.0,
    temperature_MeV: Optional[float] = None,
    cutoff_MeV: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = True,
    binary: Optional[str] = None,
    timeout: int = 600,
) -> RunResult:
    """
    Run a proton radiography simulation and return parsed results.

    Parameters
    ----------
    field
        Path to a .bfld file **or** a GridField object (saved to a temp file).
    source
        Source type: ``"parallel"``, ``"point"``, ``"disk"``, or ``"pencil"``.
    energy_MeV
        Proton kinetic energy in MeV.
    n_particles
        Number of protons to trace.
    beam_radius_mm
        Beam radius at the source plane (parallel/disk sources).
    source_distance_mm
        Distance from origin to source plane along beam axis (mm).
    angular_spread_deg
        Half-angle spread for point sources (degrees).
    beam_direction
        Unit vector along the beam axis. Default ``(1, 0, 0)``.
    detector_distance_mm
        Distance from origin to detector plane along beam axis (mm).
    detector_size_mm
        (width, height) of the detector in mm.
    detector_up
        Up vector for the detector basis. Default ``(0, 1, 0)``.
    dt_ps
        Boris integrator timestep in picoseconds.
    max_steps
        Maximum integration steps per particle.
    scale_B / scale_E
        Scale factors applied to B / E field after loading.
    energy_spread_percent
        Gaussian σ as a percentage of ``energy_MeV`` (e.g. ``5.0`` → 5 % FWHM/2.35).
        Ignored when ``temperature_MeV`` is set.
    temperature_MeV
        Exponential/TNSA spectrum temperature [MeV]: dN/dE ∝ exp(−E / T).
        When set, overrides ``energy_spread_percent``.
    cutoff_MeV
        Hard cutoff energy [MeV] for the exponential spectrum.
        Default: 100 × temperature_MeV.
    output_dir
        Where to write the run directory. If None a temp directory is used;
        files persist until you delete ``result.run_dir`` manually.
    overwrite
        Overwrite existing output directory. Default True.
    binary
        Explicit path to the proton_tracer binary (optional).
    timeout
        Subprocess timeout in seconds. Default 600.

    Returns
    -------
    RunResult
        Parsed run directory. ``raw_counts`` is always 1024×1024 (GPU native).
    """
    from ._binary import find_binary, run_binary
    from ._deck import build_deck

    bin_path = find_binary(binary)

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        # ── resolve field path ────────────────────────────────────────────────
        if isinstance(field, GridField):
            field_file = tmp / "field.bfld"
            field.save(field_file)
            field_path_str = str(field_file)
        else:
            field_path_str = str(Path(field).resolve())

        # ── write deck ───────────────────────────────────────────────────────
        deck_toml = build_deck(
            field_path_str,
            source=source,
            energy_MeV=energy_MeV,
            n_particles=n_particles,
            beam_radius_mm=beam_radius_mm,
            source_distance_mm=source_distance_mm,
            angular_spread_deg=angular_spread_deg,
            beam_direction=beam_direction,
            detector_distance_mm=detector_distance_mm,
            detector_size_mm=detector_size_mm,
            detector_up=detector_up,
            dt_ps=dt_ps,
            max_steps=max_steps,
            scale_B=scale_B,
            scale_E=scale_E,
            energy_spread_percent=energy_spread_percent,
            temperature_MeV=temperature_MeV,
            cutoff_MeV=cutoff_MeV,
        )
        deck_path = tmp / "deck.toml"
        deck_path.write_text(deck_toml)

        # ── output directory ─────────────────────────────────────────────────
        if output_dir is None:
            run_dir = Path(tempfile.mkdtemp(prefix="proton_tracer_"))
        else:
            run_dir = Path(output_dir)

        # ── invoke binary ────────────────────────────────────────────────────
        args = [str(bin_path), "run", str(deck_path), "-o", str(run_dir)]
        if overwrite:
            args.append("--overwrite")

        run_binary(args, timeout=timeout)

    return RunResult(run_dir)
