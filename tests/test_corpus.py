"""Corpus building: deterministic, parallel-safe, and a no-op when current."""

import json
import os
import pathlib

import numpy as np
import pytest

from ucsa.training import corpus, shards


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[1 + ord(c) % 50 for c in t] for t in texts]}


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(
        corpus.transformers.AutoTokenizer,
        "from_pretrained",
        lambda name: FakeTokenizer(),
    )


def write(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_documents_group_consecutive_non_empty_lines(tmp_path):
    lines = ["a", "", "b", "  ", *["c"] * corpus.LINES_PER_DOCUMENT]
    docs = list(corpus.documents(write(tmp_path, "t.txt", lines)))
    assert docs[0] == "a\nb\n" + "\n".join(
        ["c"] * (corpus.LINES_PER_DOCUMENT - 2)
    )
    assert sum(d.count("\n") + 1 for d in docs) == corpus.LINES_PER_DOCUMENT + 2


def test_each_document_ends_with_the_end_of_text_id(tmp_path):
    path = write(tmp_path, "t.txt", ["ab", "cd"])
    ids = corpus.tokenize_file(path)
    assert ids.dtype == shards.DTYPE
    assert ids[-1] == shards.EOS_ID
    assert int((ids == shards.EOS_ID).sum()) == 1


def test_an_empty_file_gives_an_empty_shard(tmp_path):
    path = write(tmp_path, "e.txt", [])
    assert corpus.tokenize_file(path).size == 0


def test_the_shard_is_the_sources_joined_in_order(tmp_path):
    a = write(tmp_path, "a.txt", ["aa"])
    b = write(tmp_path, "b.txt", ["bb"])
    out = str(tmp_path / "s.bin")
    tokens, built = corpus.build([a, b], out, workers=1)
    assert built
    expected = np.concatenate(
        [corpus.tokenize_file(a), corpus.tokenize_file(b)]
    )
    assert tokens == expected.size
    assert (np.fromfile(out, dtype=shards.DTYPE) == expected).all()


def test_rebuilding_an_unchanged_corpus_does_nothing(tmp_path):
    a = write(tmp_path, "a.txt", ["aa"])
    out = str(tmp_path / "s.bin")
    first = corpus.build([a], out, workers=1)
    stamp = os.stat(out).st_mtime_ns
    second = corpus.build([a], out, workers=1)
    assert first == (second[0], True)
    assert second[1] is False
    assert os.stat(out).st_mtime_ns == stamp


def test_a_changed_source_forces_a_rebuild(tmp_path):
    a = write(tmp_path, "a.txt", ["aa"])
    out = str(tmp_path / "s.bin")
    corpus.build([a], out, workers=1)
    write(tmp_path, "a.txt", ["aa", "longer"])
    tokens, built = corpus.build([a], out, workers=1)
    assert built
    assert tokens == corpus.tokenize_file(a).size


def test_a_different_tokenizer_forces_a_rebuild(tmp_path):
    a = write(tmp_path, "a.txt", ["aa"])
    out = str(tmp_path / "s.bin")
    corpus.build([a], out, tokenizer_name="gpt2", workers=1)
    assert corpus.build([a], out, tokenizer_name="other", workers=1)[1]


def test_a_truncated_shard_is_detected_and_rebuilt(tmp_path):
    a = write(tmp_path, "a.txt", ["aaaa"])
    out = str(tmp_path / "s.bin")
    corpus.build([a], out, workers=1)
    with open(out, "r+b") as f:
        f.truncate(2)
    assert not corpus.is_current(out, [a], "gpt2")
    assert corpus.build([a], out, workers=1)[1]


def test_a_corrupt_manifest_is_a_rebuild_not_a_crash(tmp_path):
    a = write(tmp_path, "a.txt", ["aa"])
    out = str(tmp_path / "s.bin")
    corpus.build([a], out, workers=1)
    with open(corpus.manifest_path(out), "w") as f:
        f.write("{not json")
    assert corpus.build([a], out, workers=1)[1]


def test_force_rebuilds_a_current_shard(tmp_path):
    a = write(tmp_path, "a.txt", ["aa"])
    out = str(tmp_path / "s.bin")
    corpus.build([a], out, workers=1)
    assert corpus.build([a], out, workers=1, force=True)[1]


def test_the_manifest_records_inputs_and_size(tmp_path):
    a = write(tmp_path, "a.txt", ["aa"])
    out = str(tmp_path / "s.bin")
    tokens, _ = corpus.build([a], out, workers=1)
    with open(corpus.manifest_path(out)) as f:
        manifest = json.load(f)
    assert manifest["tokens"] == tokens
    assert manifest["sources"] == [
        {"name": "a.txt", "bytes": os.path.getsize(a)}
    ]


def test_worker_processes_produce_the_same_shard_as_one_process(
    tmp_path, monkeypatch
):
    """The pool must not change the output: order and content are fixed."""
    monkeypatch.undo()
    try:
        corpus.transformers.AutoTokenizer.from_pretrained("gpt2")
    except OSError:
        pytest.skip("the gpt2 tokenizer is not available offline")
    sources = [
        write(tmp_path, f"{i}.txt", [f"the cat number {i} sat down"] * (40 + i))
        for i in range(4)
    ]
    serial = str(tmp_path / "serial.bin")
    parallel = str(tmp_path / "parallel.bin")
    corpus.build(sources, serial, workers=1)
    corpus.build(sources, parallel, workers=4)
    assert (
        pathlib.Path(serial).read_bytes() == pathlib.Path(parallel).read_bytes()
    )
