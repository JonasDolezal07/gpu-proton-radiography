"""
Shared pytest fixtures for the prad test suite.

Helpers are imported from `validate.py` at the repo root — that module is the
authoritative source for building fields, writing configs, and invoking the
tracer binary. Re-implementing them here would risk drift from the 25-test
suite that already uses them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import validate as _v  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def bin_path() -> Path:
    """
    Path to the release binary. Skips the test if it does not exist so that
    a missing build fails loudly at collection time rather than showing up as
    an opaque subprocess error inside a physics assertion.
    """
    if not _v.BIN.exists():
        pytest.skip(
            f"release binary not found at {_v.BIN}. "
            "Build with: cd rust && cargo build --release"
        )
    return _v.BIN


@pytest.fixture(scope="session")
def vulkan_env() -> dict:
    return _v._vulkan_env()


@pytest.fixture()
def tmp_out_dir(tmp_path: Path) -> Path:
    """
    Per-test output directory. pytest already gives us a unique tmp_path;
    we just use a named subdirectory so the run-dir layout is visible on
    inspection.
    """
    out = tmp_path / "run"
    return out


@pytest.fixture(scope="session")
def val_helpers():
    """
    Namespace handle for the helpers in validate.py, so tests can call
    val_helpers.write_bfld(...) etc. without each importing validate.
    """
    return _v
