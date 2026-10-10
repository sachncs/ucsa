import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reproduce.sh"


def test_the_reproduction_script_is_valid_shell():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_every_script_it_calls_exists():
    text = SCRIPT.read_text()
    called = set(re.findall(r"scripts/([a-z_]+\.py)", text))
    assert called  # it does call things
    for name in called:
        assert (ROOT / "scripts" / name).exists(), name


def test_the_dry_run_gates_training():
    """With `set -e` a failing dry run stops the script before training."""
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert text.index("dry_run.py") < text.index("train_r.py")


def test_training_is_resumable_by_default():
    assert "--resume" in SCRIPT.read_text()


def test_the_script_is_executable():
    assert SCRIPT.stat().st_mode & 0o111
