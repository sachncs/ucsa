"""One floating-point type, enforced two ways.

Static: no source file may convert between floating types, use autocast or a
loss scaler. Runtime: every floating tensor produced by a real forward,
backward and optimiser step must be in the library dtype, for both the
original model and UCSA-R.
"""

import ast
import pathlib

import pytest
import torch

from ucsa import dryrun
from ucsa.models import architecture, recurrent
from ucsa.training import engine
from ucsa.utils import precision

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = sorted(
    p
    for folder in ("ucsa", "scripts")
    for p in (ROOT / folder).rglob("*.py")
    if p.name != "precision.py" and "__pycache__" not in p.parts
)
CAST_METHODS = {"float", "double", "half", "bfloat16", "type_as"}
FLOAT_DTYPES = {"float16", "float64", "bfloat16", "half", "double", "float32"}
BANNED_NAMES = {"autocast", "GradScaler"}


def violations(source: str) -> list[tuple[int, str]]:
    """Returns `(line, what)` for every float-to-float conversion."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in CAST_METHODS and not node.args:
                found.append((node.lineno, f".{node.func.attr}()"))
            if node.func.attr in CAST_METHODS - {"float"} and node.args:
                found.append((node.lineno, f".{node.func.attr}(...)"))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"to", "type"}
        ):
            keyword = any(k.arg == "dtype" for k in node.keywords)
            positional = any(
                isinstance(a, ast.Attribute)
                and isinstance(a.value, ast.Name)
                and a.value.id in {"torch", "precision"}
                for a in node.args
            )
            if keyword or positional:
                found.append((node.lineno, ".to(dtype)"))
        if isinstance(node, ast.Attribute):
            if node.attr in BANNED_NAMES:
                found.append((node.lineno, node.attr))
            if (
                node.attr in FLOAT_DTYPES
                and isinstance(node.value, ast.Name)
                and node.value.id in {"torch", "np", "numpy"}
            ):
                found.append((node.lineno, f"{node.value.id}.{node.attr}"))
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            found.append((node.lineno, node.id))
    return sorted(set(found))


def test_the_detector_flags_every_kind_of_cast():
    src = (
        "import torch\n"
        "a = x.float()\n"
        "b = x.double()\n"
        "c = x.to(torch.float16)\n"
        "with torch.autocast('cpu'):\n"
        "    pass\n"
        "s = GradScaler()\n"
        "d = torch.zeros(1, dtype=torch.float64)\n"
    )
    flagged = {what for _, what in violations(src)}
    assert {
        ".float()",
        ".double()",
        "torch.float16",
        "autocast",
        "GradScaler",
        "torch.float64",
    } <= flagged


def test_the_detector_flags_dtype_arguments_to_to_even_across_lines():
    src = (
        "a = x.to(\n    device=d, dtype=y.dtype\n)\n"
        "b = x.to(torch.float16)\n"
        "c = x.to(device=d)\n"
    )
    assert [what for _, what in violations(src)].count(".to(dtype)") == 2


def test_the_detector_allows_exact_integer_conversions():
    assert violations("y = x.long()\nz = precision.to_dtype(mask)\n") == []


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_source_file_converts_between_floating_types(path):
    assert violations(path.read_text()) == [], path


# -------------------------------------------------------------- runtime audit


def tiny_recurrent(**kw):
    precision.configure()
    torch.manual_seed(0)
    return recurrent.Model(
        recurrent.Config(
            vocab_size=64,
            hidden=32,
            layers=2,
            heads=2,
            ffn_dim=64,
            chunk_size=8,
            banks=(("working", 4), ("long_term", 2)),
            bank_write_bias=(("working", 0.0), ("long_term", -2.0)),
            **kw,
        )
    )


@pytest.mark.parametrize(
    "extra",
    [{}, {"surprise_gate": True}, {"use_state": False}, {"slot_dropout": 1.0}],
    ids=["default", "surprise", "no-state", "slot-dropout"],
)
def test_recurrent_training_step_stays_in_one_dtype(extra):
    model = tiny_recurrent(**extra).train()
    cfg = engine.Config(steps=1, batch_size=2, seq_len=32)
    optimizer = engine.build_optimizer(model, cfg)
    x = torch.randint(0, 64, (2, 32))
    with dryrun.Recorder() as rec:
        loss, _ = model.compute_loss(x, torch.roll(x, -1, 1))
        loss.backward()
        engine.clip_grad_norm(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        model.update_ema()
    assert rec.offenders == []
    assert rec.seen <= {precision.DTYPE}
    assert dryrun.off_dtype_parameters(model) == []
    assert loss.dtype == precision.DTYPE


def test_evaluation_and_generation_stay_in_one_dtype():
    model = tiny_recurrent().eval()
    x = torch.randint(0, 64, (1, 24))
    with dryrun.Recorder() as rec:
        model(x)
        model.generate(x[:, :5], 12, temperature=0.0)
    assert rec.offenders == []


def test_original_model_forward_backward_stays_in_one_dtype():
    precision.configure()
    torch.manual_seed(0)
    model = architecture.UCSA(
        architecture.Config(hidden_size=32, vocab_size=100, num_layers=2)
    ).train()
    x = torch.randint(0, 100, (1, 8))
    with dryrun.Recorder() as rec:
        out = model(x)
        out["language"].sum().backward()
    assert rec.offenders == []
    assert dryrun.off_dtype_parameters(model) == []
    assert out["language"].dtype == precision.DTYPE


FP16_PROBE = """
import torch
from ucsa import dryrun
from ucsa.models import recurrent
from ucsa.training import engine
from ucsa.utils import precision

precision.configure()
torch.manual_seed(0)
model = recurrent.Model(recurrent.Config(
    vocab_size=64, hidden=32, layers=2, heads=2, ffn_dim=64, chunk_size=8,
    banks=(("working", 4),), bank_write_bias=(("working", 0.0),),
)).train()
cfg = engine.Config(steps=1, batch_size=2, seq_len=32)
opt = engine.build_optimizer(model, cfg)
x = torch.randint(0, 64, (2, 32))
with dryrun.Recorder(ignore_scalars=True) as rec:
    loss, _ = model.compute_loss(x, torch.roll(x, -1, 1))
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    engine.clip_grad_norm(params, 1.0)
    opt.step()
print(
    precision.DTYPE,
    dryrun.off_dtype_parameters(model),
    sorted(str(d) for d in rec.seen),
    bool(torch.isfinite(loss)),
)
"""


@pytest.mark.parametrize("name", ["float32", "float16"])
def test_one_constant_switches_the_whole_library(name):
    """The same training step, run in a fresh process under each dtype, keeps
    every parameter, buffer and gradient in that dtype with a finite loss.

    In float16 the audit also reports float32 results from inside PyTorch's
    fused kernels (RMSNorm, attention); those are not conversions in our code
    and are surfaced by the dry run instead of asserted away here. In float32
    nothing else can appear, which the other tests in this file check.
    """
    import os
    import subprocess
    import sys

    env = {**os.environ, "UCSA_DTYPE": name, "PYTHONPATH": str(ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", FP16_PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-600:]
    line = out.stdout.strip().splitlines()[-1]
    assert line.startswith(f"torch.{name} []"), line  # storage is uniform
    assert line.endswith("True"), line  # finite loss
    if name == "float32":
        assert "float16" not in line


TRAINING_ENTRY_POINTS = [
    "ucsa/training/engine.py",
    "ucsa/models/recurrent.py",
    "ucsa/training/shards.py",
    "scripts/train_r.py",
]


@pytest.mark.parametrize("path", TRAINING_ENTRY_POINTS)
def test_training_code_never_imports_the_dry_run_tooling(path):
    """Real training carries no instrumentation: auditing lives in a dry run."""
    tree = ast.parse((ROOT / path).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("dryrun" in name for name in imported), path


def test_importing_the_training_script_does_not_load_the_auditor():
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'scripts'); import train_r, "
        "ucsa.training.engine; "
        "print('ucsa.dryrun' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert out.stdout.strip().endswith("False"), out.stderr[-400:]
