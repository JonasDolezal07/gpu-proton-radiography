"""RunResult — parse a completed run directory."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np


class RunResult:
    """
    Parsed output of a single proton_tracer run.

    All arrays are loaded lazily on first access.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        if not (self.run_dir / "metadata.json").is_file():
            raise FileNotFoundError(f"Not a valid run directory: {run_dir}")
        self._metadata: Optional[dict] = None
        self._raw_counts: Optional[np.ndarray] = None
        self._processed_counts: Optional[np.ndarray] = None
        self._image = None  # PIL.Image, loaded lazily

    # ── lazy properties ───────────────────────────────────────────────────────

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            with open(self.run_dir / "metadata.json") as f:
                self._metadata = json.load(f)
        return self._metadata

    @property
    def diagnostics(self) -> dict:
        return self.metadata.get("diagnostics", {})

    @property
    def raw_counts(self) -> np.ndarray:
        if self._raw_counts is None:
            self._raw_counts = self._load_counts("raw_counts.bin", "<u4")
        return self._raw_counts

    @property
    def processed_counts(self) -> Optional[np.ndarray]:
        p = self.run_dir / "counts" / "processed_counts.bin"
        if not p.is_file():
            return None
        if self._processed_counts is None:
            self._processed_counts = self._load_counts("processed_counts.bin", "<f4")
        return self._processed_counts

    @property
    def image(self):
        """PIL.Image of the radiograph PNG, or None if not written."""
        png = self.run_dir / "images" / "radiograph.png"
        if not png.is_file():
            return None
        if self._image is None:
            from PIL import Image
            self._image = Image.open(png).copy()
        return self._image

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_counts(self, filename: str, dtype: str) -> np.ndarray:
        path = self.run_dir / "counts" / filename
        data = np.frombuffer(path.read_bytes(), dtype=dtype)
        fmt = self.metadata.get("counts_format", {})
        shape_key = "raw" if "raw" in filename else "processed"
        shape = fmt.get(shape_key, {}).get("shape")
        if shape:
            data = data.reshape(shape[0], shape[1])
        return data.copy()

    def show(self, title: Optional[str] = None) -> None:
        """Display the radiograph using matplotlib."""
        import matplotlib.pyplot as plt

        img = self.image
        if img is not None:
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis("off")
            if title:
                plt.title(title)
            plt.tight_layout()
            plt.show()
            return

        # Fallback: render raw counts with log scale
        counts = self.raw_counts.astype(np.float32)
        counts = np.log1p(counts)
        plt.figure(figsize=(6, 6))
        plt.imshow(counts, cmap="inferno", origin="lower")
        plt.colorbar(label="log(1 + counts)")
        plt.axis("off")
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()

    def save(self, path: str) -> None:
        """Save the radiograph PNG to `path`."""
        img = self.image
        if img is None:
            raise FileNotFoundError("No radiograph.png in this run directory")
        img.save(path)

    def __repr__(self) -> str:
        diag = self.diagnostics
        hits = diag.get("n_hits", "?")
        frac = diag.get("hit_fraction", "?")
        return f"<RunResult run_dir={self.run_dir.name!r} hits={hits} hit_fraction={frac}>"
