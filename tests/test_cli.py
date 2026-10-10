import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = [
    "prepare_data",
    "dry_run",
    "train_r",
    "eval",
    "compress",
    "probe_state",
    "ablate",
    "ladder",
    "tune",
    "profile_r",
    "report",
]


@pytest.mark.parametrize("tool", TOOLS)
def test_every_command_line_tool_starts_and_documents_itself(tool):
    """A tool that fails to import would otherwise be found hours into a run."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / f"{tool}.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "usage" in result.stdout.lower()


def test_the_dry_run_gate_fails_loudly_on_a_bad_override():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_r.py"),
            "--dry-run",
            "--set",
            "model.hiden=3",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )
    assert result.returncode != 0
    assert "unknown config keys" in result.stderr
