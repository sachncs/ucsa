"""Zero-shot multiple-choice benchmarks, following lm-evaluation-harness.

Tasks: HellaSwag, ARC-Easy, ARC-Challenge, PIQA and WinoGrande. Prompts and
metrics follow EleutherAI's lm-evaluation-harness so results can be compared
with published numbers: each choice is scored by its log-likelihood given the
context, `acc` ranks by total log-likelihood and `acc_norm` by
log-likelihood per character of the choice. HellaSwag and ARC-Challenge
report `acc_norm`; the other tasks report `acc`.

Loaders stream from Hugging Face `datasets` through a fixed-seed shuffle, so
a `max_examples` cap selects the same examples on every run; with no cap the
full split is evaluated. The seed is recorded on every result.
"""

import dataclasses
import math
import random
import re
from collections.abc import Callable, Iterable
from typing import Any

import datasets
import torch
import transformers

from ucsa.models import ucsa
from ucsa.training import prefix

# Seed shared by every task loader, so a `max_examples` cap picks the same
# subset on every run and for every model.
DEFAULT_EVAL_SEED = 1234

Example = dict[str, Any]


@dataclasses.dataclass
class TaskSpec:
    """One benchmark task.

    Attributes:
      name: Key in `TASK_REGISTRY`.
      loader: Called with the seed; yields examples with `context`,
        `choices`, `label` and optionally `contexts` (one per choice, for
        tasks whose context depends on the choice).
      max_examples: Cap on examples; None evaluates the full split.
      seed: Seed for the loader's shuffle.
      metric: Headline metric, `acc` or `acc_norm`. Both are always
        recorded in `EvalResult.extras`.
    """

    name: str
    loader: Callable[..., Iterable[Example]]
    max_examples: int | None = None
    seed: int = DEFAULT_EVAL_SEED
    metric: str = "acc"


@dataclasses.dataclass
class EvalResult:
    """Result of one task evaluation.

    Attributes:
      name: Task name.
      n: Examples scored.
      correct: Correct predictions under the headline metric.
      accuracy: Headline accuracy in `[0, 1]`.
      log_likelihood_mean: Mean per-token log-likelihood of the chosen
        answer.
      extras: Seed, cap, metric name, both accuracies and the binomial
        standard error of the headline metric.
    """

    name: str
    n: int
    correct: int
    accuracy: float
    log_likelihood_mean: float = 0.0
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns the result as a plain dict."""
        return dataclasses.asdict(self)


def shuffled_stream(
    ds: Iterable[Example], seed: int, buffer_size: int = 2000
) -> Iterable[Example]:
    """Yields `ds` in a deterministic shuffled order.

    Args:
      ds: A streaming Hugging Face dataset (has `.shuffle`) or any iterable.
      seed: Shuffle seed.
      buffer_size: Shuffle buffer for streaming datasets.

    Yields:
      The records of `ds`.
    """
    if hasattr(ds, "shuffle"):
        yield from ds.shuffle(seed=seed, buffer_size=buffer_size)
        return
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    yield from rows


def clean_hellaswag(text: str) -> str:
    """Removes WikiHow markup the way lm-evaluation-harness does."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag(seed: int = DEFAULT_EVAL_SEED) -> Iterable[Example]:
    """Yields HellaSwag validation examples in harness format."""
    ds = datasets.load_dataset(
        "Rowan/hellaswag", split="validation", streaming=True
    )
    for ex in shuffled_stream(ds, seed):
        context = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        yield {
            "context": clean_hellaswag(ex["activity_label"] + ": " + context),
            "choices": [clean_hellaswag(e) for e in ex["endings"]],
            "label": int(ex["label"]),
        }


def load_arc(name: str, seed: int = DEFAULT_EVAL_SEED) -> Iterable[Example]:
    """Yields ARC test examples in harness format.

    Args:
      name: `ARC-Easy` or `ARC-Challenge`.
      seed: Shuffle seed.

    Yields:
      Examples with a `Question: ... Answer:` context.
    """
    ds = datasets.load_dataset(
        "allenai/ai2_arc", name, split="test", streaming=True
    )
    for ex in shuffled_stream(ds, seed):
        yield {
            "context": "Question: " + ex["question"] + "\nAnswer:",
            "choices": ex["choices"]["text"],
            "label": ex["choices"]["label"].index(ex["answerKey"]),
        }


def load_piqa(seed: int = DEFAULT_EVAL_SEED) -> Iterable[Example]:
    """Yields PIQA validation examples in harness format.

    The upstream `ybisk/piqa` is a loading script that current `datasets`
    versions no longer run; `gimmaru/piqa` mirrors the same schema.
    """
    ds = datasets.load_dataset(
        "gimmaru/piqa", split="validation", streaming=True
    )
    for ex in shuffled_stream(ds, seed):
        yield {
            "context": "Question: " + ex["goal"] + "\nAnswer:",
            "choices": [ex["sol1"], ex["sol2"]],
            "label": int(ex["label"]),
        }


def load_winogrande(seed: int = DEFAULT_EVAL_SEED) -> Iterable[Example]:
    """Yields WinoGrande validation examples with partial scoring.

    As in lm-evaluation-harness, each option is substituted for the blank in
    the *context*, and the text after the blank is the continuation that is
    scored. The continuation is the same for both options, so the model is
    asked which prefix makes the rest of the sentence more likely.
    """
    ds = datasets.load_dataset(
        "allenai/winogrande",
        "winogrande_xl",
        split="validation",
        streaming=True,
    )
    for ex in shuffled_stream(ds, seed):
        head, _, tail = ex["sentence"].partition("_")
        yield {
            "context": "",
            "contexts": [head + ex["option1"], head + ex["option2"]],
            "choices": [tail.strip(), tail.strip()],
            "label": int(ex["answer"]) - 1,
        }


TASK_REGISTRY: dict[str, TaskSpec] = {
    "hellaswag": TaskSpec("hellaswag", load_hellaswag, metric="acc_norm"),
    "arc_easy": TaskSpec(
        "arc_easy", lambda seed=DEFAULT_EVAL_SEED: load_arc("ARC-Easy", seed)
    ),
    "arc_challenge": TaskSpec(
        "arc_challenge",
        lambda seed=DEFAULT_EVAL_SEED: load_arc("ARC-Challenge", seed),
        metric="acc_norm",
    ),
    "piqa": TaskSpec("piqa", load_piqa),
    "winogrande": TaskSpec("winogrande", load_winogrande),
}


def encode(tokenizer: Any, text: str) -> list[int]:
    """Returns token ids as a plain list.

    Args:
      tokenizer: A Hugging Face tokenizer or a wrapper exposing one as
        `.tokenizer`.
      text: Text to encode.

    Returns:
      Token ids without special tokens when the tokenizer supports that.
    """
    raw = getattr(tokenizer, "tokenizer", tokenizer)
    try:
        ids = raw.encode(text, add_special_tokens=False)
    except TypeError:
        ids = raw.encode(text)
    return [int(i) for i in ids]


def choice_loglik(
    model: Any,
    tokenizer: transformers.PreTrainedTokenizerBase,
    context: str,
    choice: str,
    device: torch.device,
    max_len: int = 1024,
) -> tuple[float, int]:
    """Scores `choice` given `context`.

    A slot model (`ucsa.UCSA`) reads only the context and its slot `j`
    predicts choice token `j`, so the choice is never in its input. A causal
    model reads context plus choice and is scored on the same tokens. At most
    `prefix.DEFAULT_NUM_TARGETS` choice tokens are scored in both cases.

    Args:
      model: Slot model or causal model.
      tokenizer: Tokenizer for the model.
      context: Conditioning text; may be empty.
      choice: Continuation to score.
      device: Device holding the model.
      max_len: Maximum input length; the start of the context is dropped.

    Returns:
      `(sum_log_prob, n_scored_tokens)`.
    """
    cont_ids = encode(tokenizer, " " + choice if context else choice)
    cont_ids = cont_ids[: prefix.DEFAULT_NUM_TARGETS]
    if not cont_ids:
        return 0.0, 0
    ctx_ids = encode(tokenizer, context) if context else []
    if not ctx_ids:
        ctx_ids = [getattr(tokenizer, "eos_token_id", None) or 0]
    cont = torch.tensor([cont_ids], dtype=torch.long, device=device)
    ctx = ctx_ids[-(max_len - len(cont_ids)) :]
    ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
    with torch.no_grad():
        if isinstance(model, ucsa.UCSA):
            out = model(ctx_t)
            logits = out.get("language", out.get("logits"))
            if logits is None:
                return 0.0, 0
            logprobs = prefix.slot_continuation_logprobs(logits, cont)
        else:
            full = torch.cat([ctx_t, cont], dim=1)
            out = model(full)
            if isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out[0] if isinstance(out, tuple) else out
            logprobs = prefix.causal_continuation_logprobs(
                logits, full, cont.shape[1]
            )
    return float(logprobs.sum().item()), int(logprobs.numel())


def conditional_loglik(
    model: Any,
    tokenizer: transformers.PreTrainedTokenizerBase,
    context: str,
    choice: str,
    device: torch.device,
    max_len: int = 1024,
) -> float:
    """Returns the mean per-token log-likelihood of `choice` given `context`.

    Args:
      model: Slot model or causal model.
      tokenizer: Tokenizer for the model.
      context: Conditioning text.
      choice: Continuation to score.
      device: Device holding the model.
      max_len: Maximum input length.

    Returns:
      The mean log-probability per scored token, or 0 if none was scored.
    """
    total, count = choice_loglik(
        model, tokenizer, context, choice, device, max_len
    )
    return total / count if count else 0.0


def evaluate_task(
    spec: TaskSpec,
    model: Any,
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: torch.device,
) -> EvalResult:
    """Runs one task.

    Args:
      spec: Task to run.
      model: Slot model or causal model.
      tokenizer: Tokenizer for the model.
      device: Device holding the model.

    Returns:
      The task's `EvalResult`.
    """
    if hasattr(model, "eval"):
        model.eval()
    correct = {"acc": 0, "acc_norm": 0}
    total = 0
    ll_sum = 0.0
    for i, ex in enumerate(spec.loader(spec.seed)):
        if spec.max_examples is not None and i >= spec.max_examples:
            break
        choices, label = ex["choices"], ex["label"]
        contexts = ex.get("contexts", [ex["context"]] * len(choices))
        scored = [
            choice_loglik(model, tokenizer, ctx, choice, device)
            for ctx, choice in zip(contexts, choices, strict=True)
        ]
        sums = [s for s, _ in scored]
        # acc ranks by total log-likelihood, acc_norm by log-likelihood per
        # character of the choice (the harness convention).
        norm = [s / max(1, len(c)) for s, c in zip(sums, choices, strict=True)]
        correct["acc"] += int(sums.index(max(sums)) == label)
        correct["acc_norm"] += int(norm.index(max(norm)) == label)
        best_sum, best_n = scored[sums.index(max(sums))]
        ll_sum += best_sum / best_n if best_n else 0.0
        total += 1
    if hasattr(model, "train"):
        model.train()
    n = max(1, total)
    acc, acc_norm = correct["acc"] / n, correct["acc_norm"] / n
    head = acc_norm if spec.metric == "acc_norm" else acc
    return EvalResult(
        name=spec.name,
        n=total,
        correct=correct[spec.metric],
        accuracy=head,
        log_likelihood_mean=ll_sum / n,
        extras={
            "seed": spec.seed,
            "max_examples": spec.max_examples,
            "metric": spec.metric,
            "acc": acc,
            "acc_norm": acc_norm,
            "stderr": math.sqrt(head * (1.0 - head) / n),
        },
    )


def evaluate_all(
    names: list[str] | None,
    model: Any,
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: torch.device,
) -> list[EvalResult]:
    """Runs several tasks.

    Args:
      names: Task names; None or empty runs every registered task. Unknown
        names are skipped with a message.
      model: Slot model or causal model.
      tokenizer: Tokenizer for the model.
      device: Device holding the model.

    Returns:
      One result per task that ran.
    """
    names = names or list(TASK_REGISTRY)
    results = []
    for name in names:
        if name not in TASK_REGISTRY:
            print(f"  unknown task {name}; skipping", flush=True)
            continue
        spec = TASK_REGISTRY[name]
        print(f"  eval {spec.name} ...", flush=True)
        result = evaluate_task(spec, model, tokenizer, device)
        results.append(result)
        print(
            f"    {result.name}: {result.correct}/{result.n} "
            f"acc={result.accuracy:.4f}",
            flush=True,
        )
    return results
