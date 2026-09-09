"""Inference entrypoint.

Run with::

    python -m ucsa.infer prompt="Once upon a time" [overrides...]

Generates up to ``inference.max_new_tokens`` new tokens using the
configured :class:`UCSA` model. Falls back to a greedy argmax when
``temperature=0`` is requested.

Optionally revises the *cause* of each token before emitting it:
``intent_steps=K`` runs K steps of gradient descent on the ``intent`` bank
alone, weights frozen, against the model's own multi-step JEPA forward model
(see :mod:`ucsa.models.intent_descent`). ``K=0`` is the default and changes
nothing. Each step costs a forward and a backward pass, so ``K=5`` is
roughly an order of magnitude more expensive per token; the reports returned
by :func:`generate_with_intent_descent` carry the forward-pass count so any
quality claim can be stated at matched compute.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from ucsa.models.intent_descent import DescentReport, optimize_intent
from ucsa.models.ucsa import UCSA, build_ucsa_from_hydra
from ucsa.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def sample_next_token(logits: Tensor, temperature: float, top_k: int) -> Tensor:
    """Pick the next token from the final-position logits.

    Args:
        logits: Tensor of shape ``(batch, vocab)``.
        temperature: Sampling temperature. ``0`` means argmax.
        top_k: Top-k filtering parameter.

    Returns:
        Token ids of shape ``(batch, 1)``.
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits / max(1e-5, temperature)
    if top_k > 0:
        values, _ = torch.topk(scaled, top_k, dim=-1)
        threshold = values[:, -1:].expand_as(scaled)
        scaled = torch.where(
            scaled < threshold,
            torch.full_like(scaled, float("-inf")),
            scaled,
        )
    return torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)


def generate_with_intent_descent(
    model: UCSA,
    prompt: Tensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    eos_token_id: int | None = None,
    intent_steps: int = 0,
    intent_learning_rate: float = 0.05,
    intent_grad_norm_threshold: float = 0.0,
    intent_grad_norm_relative_threshold: float = 0.0,
    target_encoder: torch.nn.Module | None = None,
) -> tuple[Tensor, list[DescentReport]]:
    """Generate tokens, optionally revising the origination before each one.

    Args:
        model: The UCSA model.
        prompt: Initial token ids of shape ``(batch, seq)``.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. ``0`` means argmax.
        top_k: Top-k filtering parameter.
        eos_token_id: Optional EOS token id; generation stops on EOS.
        intent_steps: ``K``, inner descent steps per token. ``0`` disables
            the inner loop entirely, which is the default.
        intent_learning_rate: Step size on the intent bank.
        intent_grad_norm_threshold: Absolute early-stop threshold on the
            intent gradient norm. ``0.0`` disables it.
        intent_grad_norm_relative_threshold: Early stop once the gradient
            norm falls to this fraction of its first-step value. This is
            the criterion that makes the step count vary per token, and so
            makes the descent's cost adaptive.
        target_encoder: Optional EMA target encoder supplying the JEPA
            targets. Strongly recommended when ``intent_steps > 0``:
            without it the objective's targets move with the origination
            being optimised.

    Returns:
        Tuple ``(tokens, reports)``. ``reports`` has one entry per token
        for which the inner loop ran, each carrying its forward-pass count
        so the cost can be reported alongside any quality claim.

    Raises:
        ValueError: If ``intent_steps`` is negative.
    """
    if intent_steps < 0:
        raise ValueError(f"intent_steps must be >= 0, got {intent_steps}.")
    model.eval()
    generated = prompt
    reports: list[DescentReport] = []
    for _ in range(max_new_tokens):
        if intent_steps > 0:
            reports.append(
                optimize_intent(
                    model,
                    generated,
                    num_steps=intent_steps,
                    learning_rate=intent_learning_rate,
                    grad_norm_threshold=intent_grad_norm_threshold,
                    grad_norm_relative_threshold=(
                        intent_grad_norm_relative_threshold
                    ),
                    target_encoder=target_encoder,
                )
            )
        with torch.no_grad():
            outputs = model(generated)
            next_token = sample_next_token(
                outputs["language"][:, -1, :], temperature, top_k
            )
        generated = torch.cat([generated, next_token], dim=-1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
    return generated, reports


def generate(
    model: UCSA,
    prompt: Tensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    eos_token_id: int | None = None,
) -> Tensor:
    """Generate ``max_new_tokens`` new tokens autoregressively.

    Args:
        model: The UCSA model.
        prompt: Initial token ids of shape ``(batch, seq)``.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. ``0`` means argmax.
        top_k: Top-k filtering parameter.
        eos_token_id: Optional EOS token id; generation stops on EOS.

    Returns:
        Tensor of shape ``(batch, prompt_len + generated)``.
    """
    model.eval()
    generated = prompt
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated)
            logits = outputs["language"][:, -1, :]
            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(1e-5, temperature)
                if top_k > 0:
                    values, _ = torch.topk(logits, top_k, dim=-1)
                    threshold = values[:, -1:].expand_as(logits)
                    logits = torch.where(
                        logits < threshold,
                        torch.full_like(logits, float("-inf")),
                        logits,
                    )
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if (
                eos_token_id is not None
                and int(next_token.item()) == eos_token_id
            ):
                break
    return generated


def run_inference(
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 42,
    intent_steps: int = 0,
    intent_learning_rate: float = 0.05,
) -> str:
    """Run end-to-end inference.

    Args:
        prompt: Input text prompt.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        seed: Random seed.
        intent_steps: ``K`` inner descent steps on the intent bank per
            token. ``0`` (default) disables the inner loop.
        intent_learning_rate: Step size on the intent bank.

    Returns:
        The generated text including the prompt.
    """
    set_seed(seed)
    model = build_ucsa_from_hydra()
    tokenizer = model.perception.tokenizer.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    eos = tokenizer.eos_token_id
    if intent_steps > 0:
        output, reports = generate_with_intent_descent(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos,
            intent_steps=intent_steps,
            intent_learning_rate=intent_learning_rate,
        )
        spent = sum(report.forward_passes for report in reports)
        LOGGER.info(
            "intent descent: %d tokens revised, %d extra forward passes",
            len(reports),
            spent,
        )
    else:
        output = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos,
        )
    return tokenizer.decode(output[0].tolist(), skip_special_tokens=True)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="UCSA inference")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--intent-steps",
        type=int,
        default=0,
        help=(
            "K steps of gradient descent on the intent bank before each "
            "token. 0 disables it. Each step costs a forward and a "
            "backward pass."
        ),
    )
    parser.add_argument("--intent-learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    text = run_inference(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        intent_steps=args.intent_steps,
        intent_learning_rate=args.intent_learning_rate,
    )
    print(text)


__all__ = [
    "generate",
    "generate_with_intent_descent",
    "main",
    "run_inference",
    "sample_next_token",
]


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
