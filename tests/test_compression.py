import math
import random

import pytest
import torch
import transformers

from ucsa.training import compression


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return transformers.AutoTokenizer.from_pretrained("gpt2")


def test_bits_per_byte_matches_hand_computation():
    # 8 bytes coded at ln(2) nats each is exactly 1 bit per byte.
    assert compression.bits_per_byte(8 * math.log(2), 8) == pytest.approx(1.0)
    assert compression.bits_per_byte(0.0, 10) == 0.0
    assert compression.bits_per_byte(1.0, 0) == math.inf


def test_byte_lengths_reproduce_the_utf8_length_of_real_text(gpt2_tokenizer):
    lengths = compression.token_byte_lengths(gpt2_tokenizer)
    for text in [
        "Hello, world!",
        "naïve café — 日本語 text with emoji 🙂",
        "line one\nline two\n\ttabbed",
    ]:
        ids = gpt2_tokenizer.encode(text)
        assert int(lengths[ids].sum()) == len(text.encode("utf-8")), text


def test_special_token_counts_as_one_byte(gpt2_tokenizer):
    lengths = compression.token_byte_lengths(gpt2_tokenizer)
    assert int(lengths[gpt2_tokenizer.eos_token_id]) == 1


def test_anchors_rank_repetitive_text_far_below_random_bytes():
    rng = random.Random(0)
    repetitive = b"the quick brown fox " * 500
    noise = bytes(rng.randrange(256) for _ in range(10_000))
    easy, hard = compression.anchors(repetitive), compression.anchors(noise)
    assert easy["zlib"] < 0.5 < 7.5 < hard["zlib"] + 0.7
    assert easy["lzma"] < hard["lzma"]
    assert hard["zlib"] > 7.8  # incompressible: no gain beyond 8 bits/byte


def test_anchors_handle_empty_input():
    assert compression.anchors(b"")["zlib"] == math.inf


def test_ideal_code_bits_equals_log2_of_inverse_probability():
    probs = torch.tensor([0.5, 0.25, 0.125])
    assert compression.ideal_code_bits(probs.log()) == pytest.approx(6.0)
