import json
import sys

import numpy as np
import pytest

sys.path.insert(0, "scripts")
import ladder  # noqa: E402

LADDER = np.array([150.0, 300.0, 600.0, 1200.0])


def curve(floor, amp, alpha):
    return floor + amp * LADDER**-alpha


def write(folder, arm, steps, loss):
    name = (
        f"{arm}.json"
        if steps == 600 and "@" not in arm
        else (f"{arm}@{steps}.json")
    )
    (folder / name).write_text(
        json.dumps(
            {
                "train_config": {"steps": steps},
                "window_nll": [loss - 0.01, loss, loss + 0.01],
            }
        )
    )


def populate(folder, arm, losses):
    for steps, loss in zip(LADDER, losses, strict=True):
        write(folder, arm, int(steps), float(loss))


def test_points_are_collected_across_folders_and_sorted(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    write(a, "base", 600, 5.0)
    write(b, "base@300", 300, 5.5)
    write(b, "base@1200", 1200, 4.6)
    write(b, "other@300", 300, 9.0)
    steps, losses = ladder.load_points([str(a), str(b)], "base")
    assert steps.tolist() == [300.0, 600.0, 1200.0]
    assert losses == pytest.approx([5.5, 5.0, 4.6])


def test_missing_folders_are_ignored(tmp_path):
    steps, _ = ladder.load_points([str(tmp_path / "nope")], "base")
    assert steps.size == 0


def test_seed_noise_is_measured_from_replicates(tmp_path):
    write(tmp_path, "base", 600, 6.00)
    write(tmp_path, "base-seed43", 600, 6.06)
    assert ladder.seed_noise([str(tmp_path)]) == pytest.approx(0.0424, abs=1e-3)


def test_seed_noise_falls_back_when_there_are_no_replicates(tmp_path):
    write(tmp_path, "base", 600, 6.0)
    assert ladder.seed_noise([str(tmp_path)]) == ladder.DEFAULT_NOISE


def test_a_genuinely_better_arm_is_called_better_at_the_target():
    data = {
        "base": (LADDER, curve(2.0, 20.0, 0.45)),
        "good": (LADDER, curve(1.7, 20.0, 0.45)),
    }
    [v] = ladder.analyze(data, 12000, noise=0.01)
    assert v.decision == "better"
    assert v.high < 0


def test_an_arm_within_noise_is_inconclusive_not_a_winner():
    data = {
        "base": (LADDER, curve(2.0, 20.0, 0.45)),
        "same": (LADDER, curve(2.0, 20.0, 0.45) + 0.002),
    }
    [v] = ladder.analyze(data, 12000, noise=0.05)
    assert v.decision == "inconclusive"


def test_an_early_leader_that_flattens_is_flagged_as_a_crossover():
    data = {
        "base": (LADDER, curve(1.0, 30.0, 0.35)),
        "fast": (LADDER, curve(2.5, 12.0, 0.6)),
    }
    [v] = ladder.analyze(data, 12000, noise=0.005)
    assert v.crossing is not None
    assert v.decision == "worse"  # it loses at the length that matters


def test_arms_with_too_few_points_are_skipped():
    data = {
        "base": (LADDER, curve(2.0, 20.0, 0.45)),
        "thin": (LADDER[:2], np.array([5.0, 4.5])),
    }
    assert ladder.analyze(data, 12000, noise=0.01) == []


def test_render_reports_bits_and_the_decision():
    data = {
        "base": (LADDER, curve(2.0, 20.0, 0.45)),
        "good": (LADDER, curve(1.7, 20.0, 0.45)),
    }
    text = ladder.render(ladder.analyze(data, 12000, 0.01), 0.01, 12000)
    assert "| good |" in text
    assert "better" in text
    assert "bits/token" in text
