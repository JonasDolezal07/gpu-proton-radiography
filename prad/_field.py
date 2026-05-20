"""GridField — .bfld read/write and numpy array wrapping."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class GridField:
    """
    B (and optional E) field on a regular 3-D grid.

    Bounds are stored in metres (SI), matching the .bfld binary format.
    Use the `bounds_m` parameter — divide mm values by 1000.
    """

    def __init__(
        self,
        B: np.ndarray,
        bounds_m: Tuple[float, float, float, float, float, float],
        *,
        E: Optional[np.ndarray] = None,
    ) -> None:
        """
        Parameters
        ----------
        B        : shape (nx, ny, nz, 3), dtype float32, Tesla
        bounds_m : (xmin, xmax, ymin, ymax, zmin, zmax) in metres
        E        : shape (nx, ny, nz, 3), dtype float32, V/m  (optional)
        """
        assert B.ndim == 4 and B.shape[3] == 3, "B must be (nx, ny, nz, 3)"
        self._B = np.ascontiguousarray(B, dtype=np.float32)
        self._bounds = tuple(float(v) for v in bounds_m)
        if E is not None:
            assert E.shape == B.shape, "E must match B shape"
            self._E: Optional[np.ndarray] = np.ascontiguousarray(E, dtype=np.float32)
        else:
            self._E = None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def data(self) -> np.ndarray:
        return self._B

    @property
    def E_data(self) -> Optional[np.ndarray]:
        return self._E

    @property
    def shape(self) -> Tuple[int, int, int]:
        nx, ny, nz, _ = self._B.shape
        return nx, ny, nz

    @property
    def bounds_m(self) -> Tuple[float, ...]:
        return self._bounds  # type: ignore[return-value]

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_array(
        cls,
        B: np.ndarray,
        bounds_m: Tuple[float, float, float, float, float, float],
        *,
        E: Optional[np.ndarray] = None,
    ) -> "GridField":
        """Create from numpy arrays. bounds_m in metres."""
        return cls(B, bounds_m, E=E)

    @classmethod
    def load(cls, path: str | Path) -> "GridField":
        """Load a .bfld file."""
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"BFLD":
                raise ValueError(f"Not a .bfld file: {path}")
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (1, 2):
                raise ValueError(f"Unsupported .bfld version {version}")
            nx, ny, nz = struct.unpack("<III", f.read(12))
            bounds = struct.unpack("<6f", f.read(24))
            f.read(64 - 4 - 4 - 12 - 24)  # reserved padding
            n = nx * ny * nz * 3
            B = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(nx, ny, nz, 3).copy()
            E = None
            if version == 2:
                E = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(nx, ny, nz, 3).copy()
        return cls(B, bounds, E=E)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write a .bfld file."""
        path = Path(path)
        version = 2 if self._E is not None else 1
        nx, ny, nz = self.shape
        with open(path, "wb") as f:
            f.write(b"BFLD")
            f.write(struct.pack("<I", version))
            f.write(struct.pack("<III", nx, ny, nz))
            f.write(struct.pack("<6f", *self._bounds))
            f.write(b"\x00" * (64 - 4 - 4 - 12 - 24))
            f.write(self._B.tobytes())
            if self._E is not None:
                f.write(self._E.tobytes())
