"""Locate the proton_tracer binary and set up the Vulkan environment for prad."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_binary(override: Optional[str] = None) -> Path:
    """
    Return the path to the proton_tracer binary.

    Search order:
      1. `override` argument (explicit path)
      2. PROTON_TRACER_BIN environment variable
      3. Bundled binary inside the installed package  (wheel install)
      4. shutil.which("proton_tracer")  (PATH)
      5. <repo>/rust/target/release/proton_tracer     (dev/repo layout)
    """
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Binary not found at explicit path: {p}")

    env_path = os.environ.get("PROTON_TRACER_BIN")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"PROTON_TRACER_BIN={env_path} does not exist")

    bundled = Path(__file__).parent / "bin" / "proton_tracer"
    if bundled.is_file():
        return bundled

    which = shutil.which("proton_tracer")
    if which:
        return Path(which)

    repo_bin = Path(__file__).parent.parent / "rust" / "target" / "release" / "proton_tracer"
    if repo_bin.is_file():
        return repo_bin

    raise FileNotFoundError(
        "proton_tracer binary not found.\n"
        "If you installed via pip, this is a bug — please report it.\n"
        "Otherwise:\n"
        "  • Set PROTON_TRACER_BIN=/path/to/proton_tracer\n"
        "  • Place proton_tracer on your PATH\n"
        "  • Build from source: cd rust && cargo build --release"
    )


def vulkan_env() -> dict:
    """Return os.environ copy with Vulkan env vars set for macOS/MoltenVK."""
    env = os.environ.copy()
    if platform.system() != "Darwin":
        return env

    icd = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    lib_dir = Path("/opt/homebrew/lib")

    if icd.exists() and "VK_ICD_FILENAMES" not in env:
        env["VK_ICD_FILENAMES"] = str(icd)

    if lib_dir.exists():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        if str(lib_dir) not in existing:
            env["DYLD_LIBRARY_PATH"] = (str(lib_dir) + ":" + existing).rstrip(":")

    return env


def run_binary(args: list[str], *, cwd: Optional[Path] = None, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run the binary, streaming stderr to the caller. Raises on non-zero exit."""
    result = subprocess.run(
        args,
        cwd=cwd,
        env=vulkan_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"proton_tracer exited with code {result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )
    return result
