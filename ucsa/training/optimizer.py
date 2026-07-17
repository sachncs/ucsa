"""Optimizers used by the UCSA trainer and the 2026 baseline.

Includes an orthogonalised-momentum SGD variant based on the algorithm
published by Keller Jordan ("Muon") and adopted by Moonshot Kimi K2
and DeepSeek V4 [DeepSeek-AI et al., "DeepSeek-V4: Towards Highly
Efficient Million-Token Context Intelligence", arXiv 2606.19348,
Apr 2026, §1.3].

Usage:
    from ucsa.training.optimizer import Muon
    opt = Muon(model.parameters(), lr=2e-2, momentum=0.95)
"""
from __future__ import annotations

import torch
from torch import Tensor


def _newton_schulz_5(G: Tensor, steps: int = 5) -> Tensor:
    """Five-iteration Newton-Schulz orthogonalisation.

    Maps a square-ish matrix toward U @ V.T (its closest orthogonal
    factor) using only matrix multiplies; this is the heart of the
    Muon update.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Orthogonalised-momentum SGD ("Muon", Keller Jordan, 2024).

    2-D parameters get a Newton-Schulz orthogonalisation of the
    momentum buffer; 1-D and 0-D parameters (biases, RMSNorm scales,
    embeddings) fall back to plain SGD with momentum — orthogonalising
    them has no geometric meaning and would slow convergence.

    Reference:
      - Keller Jordan, "Muon: An optimizer for hidden layers in
        neural networks", 2024 (blog / NanoGPT speedrun).
      - Bowne-Anderson / Raschka, "LLM Architecture in 2026", Jul
        2026 — Muon used by Moonshot and DeepSeek V4.
      - DeepSeek-AI et al., "DeepSeek-V4 ...", arXiv 2606.19348.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(group["momentum"]).add_(g)
                if g.ndim >= 2:
                    # ponytail: scale by sqrt(max(rows,cols)/cols) so the
                    # update RMS is approximately invariant to aspect ratio.
                    update = _newton_schulz_5(buf, group["ns_steps"])
                    update = update * max(
                        1.0,
                        update.size(-2) / update.size(-1),
                    ) ** 0.5
                else:
                    update = buf
                p.data.add_(update, alpha=-group["lr"])
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
        return loss


__all__ = ["Muon", "_newton_schulz_5"]
