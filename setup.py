"""
Custom wheel builder that:
  1. Compiles the Rust binary (cargo build --release)
  2. Copies it into prad/bin/
  3. Tags the wheel as platform-specific but Python-version-independent
     (py3-none-<platform>) so one wheel works with any Python 3.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).parent
BIN_SRC = ROOT / "rust" / "target" / "release" / "proton_tracer"
BIN_DST = ROOT / "prad" / "bin" / "proton_tracer"


class BuildRustAndPy(build_py):
    """build_py override: compile Rust binary before copying Python sources."""

    def run(self) -> None:
        _build_rust()
        super().run()


class PlatformWheel(bdist_wheel):
    """
    Mark the wheel as platform-specific (because it contains a compiled binary)
    but Python-version-independent (no C extension ABI).
    Result: prad-X.Y.Z-py3-none-<platform>.whl
    """

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False  # force platform tag

    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


def _build_rust() -> None:
    if BIN_DST.is_file() and not os.environ.get("FORCE_REBUILD"):
        # Skip if already compiled (dev install / re-run).
        return

    manifest = ROOT / "rust" / "Cargo.toml"
    print(f"[setup.py] cargo build --release  (manifest: {manifest})", flush=True)
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(manifest)],
        check=True,
    )

    if not BIN_SRC.is_file():
        sys.exit(f"[setup.py] ERROR: expected binary at {BIN_SRC} after cargo build")

    BIN_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BIN_SRC, BIN_DST)
    # Ensure executable bit is set (shutil.copy2 should preserve it, but be explicit).
    BIN_DST.chmod(0o755)
    print(f"[setup.py] copied binary → {BIN_DST}", flush=True)


setup(
    cmdclass={
        "build_py": BuildRustAndPy,
        "bdist_wheel": PlatformWheel,
    },
)
