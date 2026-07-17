"""Tests for the paper-table aggregator.

Only smoke tests that hit the table-rendering logic against the
file system — no model load required.
"""
from __future__ import annotations

import json
import os

from scripts.build_paper_tables import (
    _agg,
    _load_jsons,
    build_ablation_table,
    build_bank_probe_table,
    build_eval_table,
    build_main_table,
)


def _write(tmp_path, name, payload):
    p = os.path.join(str(tmp_path), name)
    with open(p, "w") as f:
        json.dump(payload, f)
    return p


def test_load_jsons_matches_a_prefix(tmp_path):
    (tmp_path / "ucsa-full-seed42.json").write_text(
        '{"best_val_ppl": 24.3, "final_val_ppl": 24.8}'
    )
    (tmp_path / "ucsa-no-ema-seed42.json").write_text(
        '{"best_val_ppl": 28.0}'
    )
    (tmp_path / "baseline.json").write_text(
        '{"final_val_ppl": 28.5}'
    )
    (tmp_path / "eval-ucsa-small.json").write_text('{"ucsa_avg_acc": 0.32}')
    matches = _load_jsons(str(tmp_path), r"^ucsa-(?!baseline).*-seed\d+\.json$")
    assert len(matches) == 2
    assert {m["name"] for m in matches} == {
        "ucsa-full-seed42.json",
        "ucsa-no-ema-seed42.json",
    }


def test_agg_single_value():
    rows = [{"data": {"best_val_ppl": 24.5}}]
    mean, sd, n = _agg(rows, "best_val_ppl")
    assert mean == 24.5
    assert sd == 0.0
    assert n == 1


def test_agg_multi_seed():
    rows = [
        {"data": {"best_val_ppl": 24.0}},
        {"data": {"best_val_ppl": 26.0}},
    ]
    mean, sd, n = _agg(rows, "best_val_ppl")
    assert mean == 25.0
    assert sd > 0.0
    assert n == 2


def test_build_main_table_with_data():
    rows = [
        {"data": {"final_val_ppl": 28.5, "best_val_ppl": 28.1}},
    ]
    out = build_main_table(
        [{"data": {"best_val_ppl": 24.3}},
         {"data": {"best_val_ppl": 25.7}}],
        rows,
    )
    # Sanity: contains the model names and the delta percentage.
    assert "Vanilla-Transformer" in out
    assert "UCSA-small" in out
    assert "%" in out


def test_build_ablation_table_groups_by_tag():
    out = build_ablation_table([
        {"name": "ucsa-full-seed42.json",
         "data": {"best_val_ppl": 24.0}},
        {"name": "ucsa-full-seed43.json",
         "data": {"best_val_ppl": 26.0}},
        {"name": "ucsa-no-ema-seed42.json",
         "data": {"best_val_ppl": 28.0}},
    ])
    assert "full" in out
    assert "no-ema" in out
    assert out.count("|") >= 4


def test_build_eval_table_pulls_each_task():
    out = build_eval_table([
        {"data": {
            "tasks": {
                "hellaswag": {"accuracy": 0.3},
                "arc_easy": {"accuracy": 0.4},
                "arc_challenge": {"accuracy": 0.2},
                "piqa": {"accuracy": 0.5},
                "winogrande": {"accuracy": 0.3},
            },
            "ucsa_avg_acc": 0.34,
        }}
    ])
    for task in ("hellaswag", "arc_easy", "arc_challenge",
                 "piqa", "winogrande"):
        assert task in out
    assert "0.3400" in out


def test_build_bank_probe_table_includes_centroid_matrix():
    out = build_bank_probe_table([
        {"data": {
            "banks": [
                {"name": "working", "norm_mean": 0.5,
                 "retention_mean": 0.3, "top_tokens": [" the"]},
                {"name": "long_term", "norm_mean": 0.5,
                 "retention_mean": 0.4, "top_tokens": [" and"]},
            ],
            "centroid_cosine_sim": {
                "names": ["working", "long_term"],
                "matrix": [[1.0, 0.5], [0.5, 1.0]],
            },
        }}
    ])
    assert "working" in out
    assert "long_term" in out
    assert "Bank-centroid" in out or "cosine" in out
