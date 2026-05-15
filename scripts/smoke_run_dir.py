#!/usr/bin/env python3
"""
Phase 2.1 smoke test — verify a full structured run produces a complete,
well-formed run directory.

Usage:
    python3 scripts/smoke_run_dir.py [--keep] [--build]

    --build   cargo build --release before running
    --keep    leave the output directory for inspection (default: delete on pass)

Exit 0 on pass, 1 on any failure.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
BIN  = ROOT / "rust/target/release/proton_tracer"


# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

_failures: list[str] = []
_warnings: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        msg = f"{label}" + (f": {detail}" if detail else "")
        print(f"  {FAIL}  {msg}")
        _failures.append(msg)
    return condition


def warn(label: str, detail: str = "") -> None:
    msg = f"{label}" + (f": {detail}" if detail else "")
    print(f"  {WARN}  {msg}")
    _warnings.append(msg)


def dtype_bytes(dtype: str) -> int:
    return {"uint32": 4, "float32": 4, "uint64": 8, "float64": 8}[dtype]


# ── test data generation ──────────────────────────────────────────────────────

def write_bfld(path: Path, B, bounds):
    nx, ny, nz = B.shape[:3]
    xmn, xmx, ymn, ymx, zmn, zmx = bounds
    with open(path, "wb") as f:
        f.write(b"BFLD")
        f.write(struct.pack("<I", 1))        # version 1 (B-only)
        f.write(struct.pack("<III", nx, ny, nz))
        f.write(struct.pack("<6f", xmn, xmx, ymn, ymx, zmn, zmx))
        f.write(b"\x00" * (64 - 4 - 4 - 12 - 24))
        f.write(B.astype("<f4").tobytes())


def write_config(path: Path, field_file: str):
    cfg = {
        "field_path": field_file,
        "detector": {
            "center_mm":  [110.0, 0.0, 0.0],
            "normal":     [1.0, 0.0, 0.0],
            "up":         [0.0, 1.0, 0.0],
            "width_mm":   500.0,
            "height_mm":  500.0,
            "pixels":     [256, 256],
        },
        "source": {
            "source_type":        "parallel",
            "n_particles":        5000,
            "energy_MeV":         14.7,
            "beam_center":        [-0.1, 0.0, 0.0],
            "beam_direction":     [1.0, 0.0, 0.0],
            "beam_radius_mm":     30.0,
            "angular_spread_deg": 0.0,
        },
        "dt_ps":     1.0,
        "max_steps": 5000,
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


# ── runner ────────────────────────────────────────────────────────────────────

def vulkan_env() -> dict:
    env = os.environ.copy()
    icd     = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    lib_dir = Path("/opt/homebrew/lib")
    if icd.exists() and "VK_ICD_FILENAMES" not in env:
        env["VK_ICD_FILENAMES"] = str(icd)
    if lib_dir.exists():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        if str(lib_dir) not in existing:
            env["DYLD_LIBRARY_PATH"] = (str(lib_dir) + ":" + existing).rstrip(":")
    return env


def run_tracer(deck: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(BIN), "run", str(deck), "-o", str(out_dir), "--overwrite"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=vulkan_env(),
        timeout=120,
    )


# ── checks ────────────────────────────────────────────────────────────────────

def check_files(run_dir: Path, meta: dict) -> None:
    print("\n  [file existence]")
    required = [
        "metadata.json",
        "resolved_config.json",
        "log.txt",
    ]
    for rel in required:
        check(rel, (run_dir / rel).is_file())

    # outputs declared in metadata must exist
    outputs = meta.get("outputs", {})
    for key, rel in outputs.items():
        if rel is not None:
            check(f"outputs.{key} → {rel}", (run_dir / rel).is_file())


def check_metadata(meta: dict) -> None:
    print("\n  [metadata fields]")
    check("metadata_schema_version == 1",
          meta.get("metadata_schema_version") == 1)

    run = meta.get("run", {})
    check("run.status == complete",  run.get("status") == "complete")
    check("run.argv non-empty",      bool(run.get("argv")))
    check("run.started_at present",  bool(run.get("started_at")))
    check("run.completed_at present", bool(run.get("completed_at")))

    code = meta.get("code", {})
    check("code.name present",    bool(code.get("name")))
    check("code.version present", bool(code.get("version")))
    if not code.get("git_commit"):
        warn("code.git_commit absent",
             "expected once initial git commit exists")

    hw = meta.get("hardware", {})
    check("hardware.gpu present",              bool(hw.get("gpu")))
    check("hardware.vulkan_api_version present", bool(hw.get("vulkan_api_version")))

    inp = meta.get("input_files", {})
    check("input_files.field_sha256 present", bool(inp.get("field_sha256")))

    check("counts_format present",  meta.get("counts_format") is not None)
    check("render present",         meta.get("render") is not None)
    check("diagnostics present",    meta.get("diagnostics") is not None)
    check("performance present",    meta.get("performance") is not None)


def check_binary_sizes(run_dir: Path, meta: dict) -> None:
    print("\n  [binary file sizes]")
    fmt = meta.get("counts_format", {})

    for kind in ("raw", "processed"):
        spec = fmt.get(kind)
        if not spec:
            continue
        shape = spec["shape"]   # [H, W]
        elem  = dtype_bytes(spec["dtype"])
        expected = shape[0] * shape[1] * elem

        # Find the file path from outputs
        key = "raw_counts" if kind == "raw" else "processed_counts"
        rel = meta.get("outputs", {}).get(key)
        if not rel:
            check(f"{kind} counts declared in outputs", False)
            continue
        p = run_dir / rel
        if p.is_file():
            actual = p.stat().st_size
            check(
                f"{kind} counts size {actual} == {expected} "
                f"(shape {shape[0]}×{shape[1]} × {elem}B)",
                actual == expected,
            )
        else:
            check(f"{kind} counts file exists", False)


def check_diagnostics(meta: dict) -> None:
    print("\n  [diagnostics]")
    diag = meta.get("diagnostics", {})
    if not diag:
        check("diagnostics block present", False)
        return
    check("n_hits > 0", diag.get("n_hits", 0) > 0,
          f"got {diag.get('n_hits')}")
    check("hit_fraction in (0, 1]",
          0 < diag.get("hit_fraction", 0) <= 1.0,
          f"got {diag.get('hit_fraction')}")


def check_log(run_dir: Path) -> None:
    print("\n  [log.txt]")
    log_path = run_dir / "log.txt"
    if not log_path.is_file():
        check("log.txt exists", False)
        return
    text = log_path.read_text(errors="replace")
    check("log.txt non-empty",              len(text) > 0)
    check("log mentions simulation complete",
          "simulation complete" in text.lower() or "complete" in text.lower())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    keep  = "--keep"  in sys.argv
    build = "--build" in sys.argv

    if build:
        print("Building proton_tracer (release)…")
        r = subprocess.run(["cargo", "build", "--release"],
                           cwd=ROOT / "rust")
        if r.returncode != 0:
            sys.exit("Build failed")
        print("Build OK\n")

    if not BIN.exists():
        print(f"Binary not found: {BIN}")
        print("Run with --build or: cd rust && cargo build --release")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="proton_smoke_"))
    run_out = tmp / "run_out"

    try:
        # --- generate test data ---
        bfld = tmp / "zero.bfld"
        B    = np.zeros((8, 8, 8, 3), dtype=np.float32)
        write_bfld(bfld, B, (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06))
        deck = tmp / "zero_straight.json"
        write_config(deck, str(bfld))

        # --- run ---
        print(f"Running: proton_tracer run {deck.name} -o run_out")
        result = run_tracer(deck, run_out)

        if result.returncode != 0:
            print(f"\n{FAIL}  Binary exited {result.returncode}")
            print("--- stderr ---")
            print(result.stderr[-3000:])
            return 1

        check("binary exited 0", True)

        # --- load metadata ---
        meta_path = run_out / "metadata.json"
        if not meta_path.is_file():
            print(f"\n{FAIL}  metadata.json not found in {run_out}")
            return 1
        meta = json.loads(meta_path.read_text())

        # --- run checks ---
        check_files(run_out, meta)
        check_metadata(meta)
        check_binary_sizes(run_out, meta)
        check_diagnostics(meta)
        check_log(run_out)

        # --- summary ---
        print()
        if _failures:
            print(f"{'─'*52}")
            print(f"  {FAIL}  {len(_failures)} check(s) failed:")
            for f in _failures:
                print(f"       • {f}")
            if _warnings:
                print(f"  {WARN}  {len(_warnings)} warning(s):")
                for w in _warnings:
                    print(f"       • {w}")
            return 1
        else:
            print(f"{'─'*52}")
            print(f"  {PASS}  All checks passed"
                  + (f"  ({len(_warnings)} warning(s))" if _warnings else ""))
            if _warnings:
                for w in _warnings:
                    print(f"  {WARN}  {w}")
            if keep:
                print(f"\n  Run directory kept at: {run_out}")
            return 0

    finally:
        if not keep or _failures:
            shutil.rmtree(tmp, ignore_errors=True)
        elif keep:
            pass  # kept above


if __name__ == "__main__":
    sys.exit(main())
