"""Files that must meet the strict engineering standard.

Kept in one place so the naming test, CI and humans agree on the scope.
"""

STRICT_FILES: tuple[str, ...] = (
    "ucsa/models/recurrent.py",
    "ucsa/models/graph.py",
    "ucsa/training/engine.py",
    "ucsa/training/shards.py",
    "ucsa/training/eval_harness.py",
    "ucsa/training/tuning.py",
    "ucsa/training/prefix.py",
    "ucsa/training/reference.py",
    "scripts/train_r.py",
    "scripts/train.py",
    "scripts/eval.py",
    "scripts/prepare_data.py",
    "scripts/small_config.py",
    "scripts/tune.py",
    "scripts/profile_r.py",
    "tests/strict_files.py",
    "tests/test_naming_policy.py",
    "tests/test_recurrent.py",
    "tests/test_graph.py",
    "tests/test_engine.py",
    "tests/test_tuning.py",
    "tests/test_prefix_protocol.py",
    "tests/test_reference.py",
    "tests/test_shards.py",
    "tests/test_eval_harness.py",
)
