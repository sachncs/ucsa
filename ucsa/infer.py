"""Inference entrypoint.

Run with::

    python -m ucsa.infer prompt="Once upon a time" [overrides...]

Generates up to ``inference.max_new_tokens`` new tokens using the
configured :class:`UCSA` model. Falls back to a greedy argmax when
``temperature=0`` is requested.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from ucsa.models.ucsa import UCSA, build_ucsa_from_hydra
from ucsa.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


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
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
    return generated


def run_inference(
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 42,
) -> str:
    """Run end-to-end inference.

    Args:
        prompt: Input text prompt.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering.
        seed: Random seed.

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
    args = parser.parse_args()
    text = run_inference(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(text)


__all__ = ["generate", "main", "run_inference"]


if __name__ == "__main__":  # pragma: no cover - script entry
    main()