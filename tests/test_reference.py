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
        "piqa": {"accuracy": 0.55, "n": 1838, "extras": {"stderr": 0.0116}},
        "unknown_task": {"accuracy": 1.0, "n": 1, "extras": {}},
    }
    rows = compare(ours, ref)
    assert [r["task"] for r in rows] == ["piqa"]
    assert rows[0]["ours"] == pytest.approx(55.0)
    assert rows[0]["Mamba-130M"] == 64.5
    assert "55.0 ± 1.2" in format_table(rows, ref)
