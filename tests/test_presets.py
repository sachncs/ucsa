"""Presets and overrides: one config file, validated before any compute."""

import json

import pytest

from ucsa.training import presets


def test_every_shipped_preset_builds_valid_configs():
    for name in presets.names():
        model, train = presets.build_configs(name)
        assert model.hidden % model.heads == 0
        assert train.steps > 0


def test_loading_returns_an_independent_copy():
    a = presets.load("small")
    a["model"]["hidden"] = 1
    assert presets.load("small")["model"]["hidden"] != 1


def test_an_unknown_preset_names_the_alternatives():
    with pytest.raises(ValueError, match="small"):
        presets.load("nope")


@pytest.mark.parametrize(
    "bad", ["nokey", "model=3", "model.=3", "optim.lr=1", "=1"]
)
def test_malformed_overrides_are_rejected(bad):
    with pytest.raises(ValueError):
        presets.apply_overrides(presets.load("tiny"), [bad])


def test_overrides_parse_json_values():
    config = presets.apply_overrides(
        presets.load("tiny"),
        ["model.read_gate=true", "train.lr=2.4e-3", "model.chunk_size=16"],
    )
    assert config["model"]["read_gate"] is True
    assert config["train"]["lr"] == pytest.approx(2.4e-3)
    assert config["model"]["chunk_size"] == 16


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("3e-4", 3e-4),
        ("2.4e-3", 2.4e-3),
        ("true", True),
        ("false", False),
        ("null", None),
        ("[1, 2]", [1, 2]),
        ("64", 64),
        ("word", "word"),
    ],
)
def test_values_parse_the_way_they_look(raw, value):
    assert presets.parse_value(raw) == value


def test_the_last_override_wins():
    config = presets.apply_overrides(
        presets.load("tiny"), ["train.lr=1e-3", "train.lr=2e-3"]
    )
    assert config["train"]["lr"] == pytest.approx(2e-3)


def test_an_unknown_field_is_rejected_by_the_config_class():
    with pytest.raises((ValueError, TypeError)):
        presets.build_configs("tiny", ["model.no_such_field=1"])


def test_explicit_overrides_beat_auto_sizing():
    sized, _ = presets.build_configs("tiny", params=5_000_000)
    pinned, _ = presets.build_configs(
        "tiny", ["model.hidden=64"], params=5_000_000
    )
    assert sized.hidden != 64
    assert pinned.hidden == 64


def test_seed_and_output_directory_reach_the_train_config():
    _, train = presets.build_configs("tiny", seed=7, out_dir="x/y")
    assert (train.seed, train.out_dir) == (7, "x/y")


def test_a_custom_file_replaces_the_default(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"presets": {"mine": {"model": {}}}}))
    assert presets.names(str(path)) == ["mine"]
    assert presets.load("mine", str(path)) == {"model": {}, "train": {}}


def test_batches_are_deterministic_and_resumable(tmp_path):
    from ucsa.training import shards

    ids = list(range(5000))
    shards.write_shard([ids], str(tmp_path / "train.bin"))
    shards.write_shard([ids], str(tmp_path / "val.bin"))
    _, train_config = presets.build_configs("tiny", ["train.batch_size=2"])
    train, val = presets.batch_factories(train_config, str(tmp_path))
    first = [x for x, _ in zip(train(0), range(4), strict=False)]
    again = [x for x, _ in zip(train(0), range(4), strict=False)]
    resumed = next(train(2))
    assert all((a[0] == b[0]).all() for a, b in zip(first, again, strict=True))
    assert (resumed[0] == first[2][0]).all()
    assert next(val(0))[0].shape[0] == 2
