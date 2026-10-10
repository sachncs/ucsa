import json
import sys

import numpy as np
import pytest

sys.path.insert(0, "scripts")
import ablate  # noqa: E402


def record(name, losses, ppl=100.0):
    return {
        "name": name,
        "params": 1_000_000,
        "tokens_per_second": 9000,
        "final": {"ppl_all": ppl, "bpb_all": 1.9},
        "window_nll": list(losses),
    }


def test_noise_is_the_spread_of_the_base_replicates():
    recs = {
        "base": record("base", [6.00] * 4),
        "base-seed43": record("base-seed43", [6.04] * 4),
        "other": record("other", [1.0] * 4),
    }
    assert ablate.replicate_noise(recs) == pytest.approx(0.0283, abs=1e-3)


def test_noise_is_unmeasured_with_a_single_base():
    assert ablate.replicate_noise({"base": record("base", [6.0])}) is None


def test_a_gap_inside_the_noise_is_not_called_a_winner():
    assert ablate.verdict_for(-0.03, True, noise=0.02) == (
        "within run-to-run noise"
    )


def test_a_gap_beyond_twice_the_noise_is_called_by_its_sign():
    assert ablate.verdict_for(-0.10, True, noise=0.02) == "better"
    assert ablate.verdict_for(+0.10, True, noise=0.02) == "worse"


def test_an_insignificant_interval_is_never_called_a_difference():
    assert ablate.verdict_for(-0.5, False, noise=0.01) == (
        "no significant difference"
    )


def test_without_replicates_a_significant_gap_is_flagged_not_trusted():
    assert ablate.verdict_for(-0.5, True, noise=None) == "noise not measured"


def test_the_summary_table_applies_the_noise_rule(tmp_path):
    rng = np.random.default_rng(0)
    base = 6.0 + rng.normal(0, 0.01, 200)
    arms = {
        "base": base,
        "base-seed43": base + 0.04,  # replicate noise ~0.028 nats
        "tiny-gain": base - 0.03,  # significant by window CI, inside noise
        "real-gain": base - 0.30,
    }
    for name, losses in arms.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(record(name, losses)))
    text = ablate.summarise(str(tmp_path))
    rows = {
        line.split("|")[1].strip(): line
        for line in text.splitlines()
        if line.startswith("|")
    }
    assert "within run-to-run noise" in rows["tiny-gain"]
    assert "better" in rows["real-gain"]
    assert "Run-to-run noise" in text


def test_a_given_noise_overrides_an_underdetermined_estimate(tmp_path):
    base = np.full(50, 6.0)
    for name, losses in {
        "base": base,
        "base-seed43": base + 0.0005,  # two replicates that happen to agree
        "arm": base - 0.015,  # 0.022 bits: > lucky noise, < 2 x 0.0145
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(record(name, losses)))
    trusting = ablate.summarise(str(tmp_path))
    careful = ablate.summarise(str(tmp_path), noise_bits=0.0145)

    def arm(text):
        return next(
            line
            for line in text.splitlines()
            if line.startswith("| arm |")
            and line.split("|")[2].strip().endswith("M")
        )

    assert "better" in arm(trusting)  # fooled by the lucky pair
    assert "within run-to-run noise" in arm(careful)
    assert "given" in careful
