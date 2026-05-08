"""
C compilation checks.

For every generated folder produced by test_generate.py, compiles main.c with
  gcc -fsyntax-only -std=c99
to catch type errors, missing symbols, and other C-level issues without
needing to link or run the binary.

Skips gracefully when:
  - gcc is not on PATH
  - the generated folder doesn't exist yet (run test_generate.py first)

Run with:  pytest tests/test_compile.py -v
"""
import os
import shutil
import subprocess

import pytest

from .conftest import GENERATED_DIR

# ---------------------------------------------------------------------------
# Same axis as test_generate so IDs match
# ---------------------------------------------------------------------------

_FW_TASK = [
    ("xgb", "regression"), ("xgb", "binary"), ("xgb", "multiclass"),
    ("lgb", "regression"), ("lgb", "binary"), ("lgb", "multiclass"),
]
_CONFIGS = ["default", "q16", "q8"]


def _gcc() -> str | None:
    """Return path to gcc (or cc), or None if not found."""
    return shutil.which("gcc") or shutil.which("cc")


@pytest.mark.parametrize("cfg_name", _CONFIGS)
@pytest.mark.parametrize("fw,task", _FW_TASK, ids=[f"{f}_{t}" for f, t in _FW_TASK])
def test_compile(fw: str, task: str, cfg_name: str) -> None:
    """Syntax-check the generated main.c with gcc -fsyntax-only."""
    gcc = _gcc()
    if gcc is None:
        pytest.skip("gcc / cc not found on PATH — install MinGW or GCC to enable compile checks")

    folder = os.path.join(GENERATED_DIR, f"{fw}_{task}_{cfg_name}")
    main_c = os.path.join(folder, "main.c")

    if not os.path.isfile(main_c):
        pytest.skip(f"Generated folder not found: {folder} — run test_generate.py first")

    result = subprocess.run(
        [gcc, "-fsyntax-only", "-std=c99", "-Wall", "main.c"],
        cwd=folder,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"[{fw} {task} {cfg_name}] gcc reported errors:\n"
        f"  stdout: {result.stdout.strip()}\n"
        f"  stderr: {result.stderr.strip()}"
    )
