"""Zero-shot benchmarks scored by log-likelihood.

BLiMP (Warstadt et al., 2020): 67 phenomena of English grammar, 1,000 minimal
pairs each. A pair is correct when the model gives the grammatical sentence a
strictly higher total log-probability than the ungrammatical one; the score is
the mean accuracy over phenomena, the number the BabyLM challenge reports. A
sentence is conditioned on the end-of-text token only, so nothing but the
sentence itself is read.

A task is a loader that yields examples with a `context`, a list of
`choices`, the `label` of the right one and an optional `group`. Loaders
shuffle with a fixed seed, so a `max_examples` cap selects the same examples
on every run; with no cap the full split is evaluated.
"""

import collections
import dataclasses
import math
import random
from collections.abc import Callable, Iterable
from typing import Any

import datasets
import torch
import transformers
from torch.nn import functional

from ucsa.models import recurrent
from ucsa.training import scoring

# Seed shared by every task loader, so a `max_examples` cap picks the same
# subset on every run and for every model.
DEFAULT_EVAL_SEED = 1234

Example = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Task:
    """One benchmark task.

    Attributes:
      name: Key in `TASKS`.
      loader: Called with the seed; yields examples.
    """

    name: str
    loader: Callable[[int], Iterable[Example]]


@dataclasses.dataclass(frozen=True)
class Result:
    """Result of one task evaluation.

    Attributes:
      name: Task name.
      n: Examples scored.
      accuracy: Mean accuracy over groups, in `[0, 1]`.
      stderr: Standard error of `accuracy`.
      groups: Accuracy of each group (for BLiMP, each phenomenon).
      seed: Shuffle seed of the loader.
      max_examples: Cap on examples; None means the full split.
    """

    name: str
    n: int
    accuracy: float
    stderr: float
    groups: dict[str, float]
    seed: int
    max_examples: int | None

    def to_dict(self) -> dict[str, Any]:
        """Returns the result as a plain dict."""
        return dataclasses.asdict(self)


def load_blimp(seed: int = DEFAULT_EVAL_SEED) -> Iterable[Example]:
    """Yields BLiMP minimal pairs, grammatical sentence first.

    Args:
      seed: Shuffle seed.

    Yields:
      Examples with an empty context, `[good, bad]` as choices, label 0 and
      the phenomenon name as `group`.
    """
    rows = []
    for name in datasets.get_dataset_config_names("nyu-mll/blimp"):
        for ex in datasets.load_dataset("nyu-mll/blimp", name, split="train"):
            rows.append(
                {
                    "context": "",
                    "choices": [ex["sentence_good"], ex["sentence_bad"]],
                    "label": 0,
                    "group": name,
                }
            )
    random.Random(seed).shuffle(rows)
    yield from rows


TASKS: dict[str, Task] = {"blimp": Task("blimp", load_blimp)}


def encode(
    tokenizer: transformers.PreTrainedTokenizerBase, text: str
) -> list[int]:
    """Returns the token ids of `text` without special tokens.

    Args:
      tokenizer: A Hugging Face tokenizer.
      text: Text to encode.

    Returns:
      Token ids as a plain list.
    """
    return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]


@torch.no_grad()
def choice_logprobs(
    model: recurrent.Model,
    tokenizer: transformers.PreTrainedTokenizerBase,
    context: str,
    choices: list[str],
    device: torch.device,
    max_len: int = 1024,
) -> list[tuple[float, int]]:
    """Scores every choice given the context, in one forward pass.

    Each choice is read after the context and scored on its own tokens, at
    most `scoring.DEFAULT_NUM_TARGETS` of them. Rows are right-padded; the
    model is causal, so padding never changes the score of an earlier token.

    Args:
      model: Causal model.
      tokenizer: Tokenizer for the model.
      context: Conditioning text; empty means the end-of-text token.
      choices: Continuations to score.
      device: Device holding the model.
      max_len: Maximum input length; the start of the context is dropped.

    Returns:
      `(sum_log_prob, n_scored_tokens)` per choice; `(0.0, 0)` for a choice
      with no tokens.
    """
    ctx_ids = encode(tokenizer, context) if context else []
    ctx_ids = ctx_ids or [tokenizer.eos_token_id or 0]
    rows: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    for choice in choices:
        cont = encode(tokenizer, " " + choice if context else choice)
        cont = cont[: scoring.DEFAULT_NUM_TARGETS]
        ctx = ctx_ids[-(max_len - len(cont)) :]
        rows.append(ctx + cont)
        spans.append((len(ctx), len(cont)))
    width = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row)
    ids = ids.to(device)
    model.eval()
    logits = model(ids)["logits"]
    out = []
    for i, (start, count) in enumerate(spans):
        if count == 0:
            out.append((0.0, 0))
            continue
        pred = logits[i, start - 1 : start - 1 + count]
        nll = functional.cross_entropy(
            pred, ids[i, start : start + count], reduction="sum"
        )
        out.append((-float(nll), count))
    return out


def evaluate_task(
    task: Task,
    model: recurrent.Model,
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: torch.device,
    max_examples: int | None = None,
    seed: int = DEFAULT_EVAL_SEED,
) -> Result:
    """Runs one task.

    An example counts as correct only when the right choice scores strictly
    higher than every other, so ties never favour the first choice.

    Args:
      task: Task to run.
      model: Causal model.
      tokenizer: Tokenizer for the model.
      device: Device holding the model.
      max_examples: Cap on examples; None evaluates the full split.
      seed: Shuffle seed of the loader.

    Returns:
      The task's `Result`.
    """
    hits: dict[str, list[int]] = collections.defaultdict(list)
    for i, ex in enumerate(task.loader(seed)):
        if max_examples is not None and i >= max_examples:
            break
        scored = choice_logprobs(
            model, tokenizer, ex["context"], ex["choices"], device
        )
        sums = [total for total, _ in scored]
        right = sums[ex["label"]]
        won = all(right > s for j, s in enumerate(sums) if j != ex["label"])
        hits[ex.get("group", "all")].append(int(won))
    groups = {g: sum(h) / len(h) for g, h in hits.items()}
    accuracy = sum(groups.values()) / len(groups) if groups else 0.0
    variance = sum(p * (1.0 - p) / len(hits[g]) for g, p in groups.items())
    return Result(
        name=task.name,
        n=sum(len(h) for h in hits.values()),
        accuracy=accuracy,
        stderr=math.sqrt(variance) / max(1, len(groups)),
        groups=groups,
        seed=seed,
        max_examples=max_examples,
    )


def evaluate_all(
    names: list[str] | None,
    model: recurrent.Model,
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: torch.device,
    max_examples: int | None = None,
) -> list[Result]:
    """Runs several tasks.

    Args:
      names: Task names; None or empty runs every task.
      model: Causal model.
      tokenizer: Tokenizer for the model.
      device: Device holding the model.
      max_examples: Cap per task; None evaluates the full splits.

    Returns:
      One result per task.

    Raises:
      ValueError: If a name is not a task.
    """
    names = names or list(TASKS)
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; have {list(TASKS)}")
    results = []
    for name in names:
        print(f"  eval {name} ...", flush=True)
        result = evaluate_task(
            TASKS[name], model, tokenizer, device, max_examples
        )
        results.append(result)
        print(f"    {name}: acc={result.accuracy:.4f} n={result.n}", flush=True)
    return results
