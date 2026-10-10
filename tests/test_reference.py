import pytest

from ucsa.training.reference import compare, format_table, load_reference


def test_reference_file_is_complete_and_sourced():
    ref = load_reference()
    assert "arXiv" in ref["source"]
    tasks = set(ref["metrics"])
    assert tasks == set(ref["chance"])
    for name, m in ref["models"].items():
        assert tasks <= set(m), name
        # Every published number must sit above a sane floor.
        for t in tasks:
            assert m[t] >= ref["chance"][t] - 3, (name, t)


def test_compare_lines_up_ours_with_references():
    ref = load_reference()
    ours = {
        "blimp": {"accuracy": 0.71, "n": 67000, "stderr": 0.0017},
        "unknown_task": {"accuracy": 1.0, "n": 1, "stderr": 0.0},
    }
    rows = compare(ours, ref)
    assert [r["task"] for r in rows] == ["blimp"]
    assert rows[0]["ours"] == pytest.approx(71.0)
    assert rows[0]["OPT-125M (baseline)"] == 75.0
    assert "71.0 ± 0.2" in format_table(rows, ref)
