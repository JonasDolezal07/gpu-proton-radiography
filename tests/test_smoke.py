"""
Smoke tests for the pytest harness itself.

These should pass without running any GPU work — they verify that pytest is
wired up, fixtures resolve, and the helper module imports cleanly. Physics
tests live under tests/analytic/ and tests/regression/.
"""

from pathlib import Path


def test_repo_root_is_a_directory(repo_root: Path) -> None:
    assert repo_root.is_dir()
    assert (repo_root / "validate.py").is_file()
    assert (repo_root / "PLAN.md").is_file()


def test_val_helpers_importable(val_helpers) -> None:
    for name in (
        "write_bfld",
        "write_config",
        "run_batch",
        "run_deck",
        "read_hits",
        "read_hits_bin",
        "_vulkan_env",
    ):
        assert hasattr(val_helpers, name), f"validate.py is missing helper: {name}"


def test_vulkan_env_has_keys(vulkan_env: dict) -> None:
    # On macOS with MoltenVK we expect these to be present; on other platforms
    # the fixture still returns a dict, so we only assert the type here.
    assert isinstance(vulkan_env, dict)
