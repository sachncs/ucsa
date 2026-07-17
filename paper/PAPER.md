# UCSA — Paper Draft

> Working title: **"UCSA: Persistent Cognitive State Anchors a
> Multi-Step JEPA Chain Across Reasoning Iterations"**
> Target venue: ICML / NeurIPS (efficient inference track or
> architecture track).
>
> This draft is under active development. Numbers in tables are
> placeholders until the ablation suite finishes running; the
> paper-promised experiments are wired into ``scripts/eval.py`` and
> ``scripts/train_baseline.py``.

---

## Abstract

We study the design question: *what happens if a foundation model's
*entire* computation orbits a single, persistent, differentiable
cognitive state (PCS)*, with everything else — language modelling,
planning, tool use, predictive world modelling — cast as projections
of that state. We pair this architectural commitment with a novel
auxiliary objective — **multi-step JEPA prediction chained through
the reasoning loop's intermediates**, with a hard-EMA target encoder
tracking latents across the chain — and an input-reconstruction
head that enforces a capacity bottleneck on the latent space.

Our contributions:

1. **PCS as the central architectural primitive.** Six token banks
   with explicit roles, retention scoring, and recycle policy.
2. **Multi-step JEPA prediction across reasoning iterations** with
   EMA-tracked targets. Lightweight, no multi-term loss juggling,
   no stop-grad gymnastics.
3. **A matched-compute evaluation suite** that brings standard LLM
   benchmarks (HellaSwag, ARC, PIQA, WinoGrande) to UCSA-scale
   training and reports numbers against a from-scratch vanilla
   Transformer trained on the same data for the same steps.

## 1. Introduction

The dominant 2024-2026 line of LLM research has been stacking improvements onto the Transformer decoder: better attention variants (MLA, GQA, sliding-window), better routing (DeepSeekMoE), better context compression (CSA/HCA), and so on. The structural commitments — there's a residual stream, an attention mechanism, and a wide MLP — stay untouched.

We invert the framing. UCSA places a single, persistent, differentiable **cognitive state** at the centre of every forward pass and operates on it with a configurable transition operator. The same state feeds a language-model head, a JEPA predictive head, an input-reconstruction head, and a memory service that ships new long-term content. The Transformer is just the current implementation of the operator — the abstraction accommodates Mamba, RWKV, or other state-space variants without changing the cognitive architecture.

The training innovation is to chain JEPA prediction across the operator's iterations and to stabilise the prediction with a hard-EMA target encoder whose targets span the same chain. The intuition: rather than asking the model to predict a single next latent, ask it to predict $k$ successive latents for $k = N_\text{iter} - 1$ where each prediction is anchored against an EMA-tracked latent.

## 2. Related Work

- **Joint-Embedding Predictive Architectures (I-JEPA / LeWM)**: JEPA
  as the auxiliary loss; LeWM §3.4 collapses I-JEPA's multi-term
  loss into one SmoothL1 + Gaussian regulariser. UCSA uses the same
  LeWM-style loss and extends it to a multi-step prediction chain.
- **Hard-EMA target encoders (DINO, I-JEPA, LeWM)**: standard
  anti-collapse machinery. We replicate the technique inside
  UCSA and use it to align the multi-step chain.
- **Persistent state in LLMs (Mamba, RWKV, Hyena, Hyena Hierarchy)**:
  recurrent state as an alternative to attention. UCSA keeps the
  Transformer operator but elevates state to *first-class* and
  retention-aware.
- **KV-cache compression (MLA, GQA, CSA/HCA)**: orthogonal.
  Operates on the encoder side; UCSA operates on the JEPA loss
  side.
- **Mixture of Experts (DeepSeekMoE, Stable LatentMoE)**: orthogonally
  applicable to the operator's FFN.

## 3. Method

### 3.1. Persistent Cognitive State (PCS)

Six token banks with explicit roles (numbers default):

| Bank            | Tokens | Role                                           |
| --------------- | -----: | ---------------------------------------------- |
| WorkingMemory   |     64 | Scratch space, mutated each iteration.          |
| LongTermMemory  |    128 | Accepted knowledge, retained across requests.  |
| Goal            |     16 | Active objective.                              |
| Episode         |     32 | Per-request context.                            |
| Task            |     16 | Long-running task state.                        |
| MemoryIndex     |     32 | Retrieval index, cross-attended each block.     |

Retention metadata drives a recycle policy: the bottom-$k$ scored
long-term tokens are recycled when new content arrives. See
``docs/architecture.md``.

### 3.2. The reasoning loop

Each forward:

1. Inject the new observation into WorkingMemory.
2. For $N$ iterations (default 4): $C' \leftarrow F(C, O)$ where $F$
   is the current transition operator (a Transformer with
   Grouped-Query Attention, RoPE, RMSNorm, optional MoE).
3. Project WorkingMemory through the four heads (language,
   planning, tool, input-reconstruct).

### 3.3. Multi-step JEPA prediction chain

The reasoning loop captures WorkingMemory after each iteration
call as a detached clone. This gives a sequence of latents

$$
z_0, z_1, \ldots, z_{N-1}
$$

We frame each consecutive pair as a JEPA prediction target:

$$
\mathcal{L}_\text{JEPA} = \frac{1}{N-1} \sum_{k=0}^{N-2}
    \text{SmoothL1}(z_k,\, \tilde{z}_{k+1})
$$

where $\tilde{z}_{k+1}$ comes from a **hard-EMA target encoder**
that tracks the live model under momentum $\mu = 0.996$. The
per-pair loss back-propagates only through the predicted $z_k$;
the EMA-tracked target is no-grad. In the LeWM-style variant, a
per-pair **Gaussian regulariser** keeps each $z_k$ near
$\mathcal{N}(0, I)$.

The chain is implemented in ``UCSA.forward`` (see
``jepa_multi_step`` in the output dict) and consumed by
``JEPALoss.forward(multi_step_pairs=...)``.

### 3.4. Input-reconstruction capacity bottleneck

A fifth projection head reads WorkingMemory and predicts
``perception.embed_tokens(inputs)`` under a sliced-to-``seq_len``
loss. Force-aligns the latent to retain enough information to
recover the input — closes the collapse gap that a one-term
SmoothL1 leaves open.

### 3.5. Stable training

- **AdamW** for baseline runs; **Muon** (orthogonalised momentum
  SGD, Keller Jordan 2024) as the on-by-default optimiser when
  the muon flag is set.
- **Cosine warmup** over the first 400 steps.
- **Hard-EMA target encoder** for the JEPA chain (momentum 0.996).
- **TC-JEPA sparse text conditioner** (arXiv 2605.03245) at
  scale 0.1 — top-k cross-attention from input token embeddings
  into each predicted latent in the chain.
- **Curriculum** gates losses by stage: language-only →
  language+JEPA → language+JEPA+memory → joint+router.

## 4. Experimental Setup

| Item              | Choice                                              |
| ----------------- | --------------------------------------------------- |
| Corpus            | fineweb-edu ``train`` split, streamed              |
| Tokenizer         | GPT-2 BPE (50,257 vocab)                           |
| Sequence length   | 1024                                                |
| Hardware          | Apple-silicon MPS (development); CUDA for paper runs |
| Seeds             | 42 primary; secondary seeds via ``--seed``          |
| Eval datasets     | HellaSwag, ARC-e, ARC-c, PIQA, WinoGrande (200-item subsets) |
| Eval protocol     | Rank-by-conditional-log-likelihood (lm-eval-harness style) |

### 4.1. Baselines

- **Vanilla-Transformer**: from-scratch GPT-2-style (no PCS hooks,
  no MoE, no JEPA, no Muon) trained on the same stream for the
  same steps. Implemented in ``scripts/train_baseline.py``.
- **Public LMs (zero-shot)**: ``gpt2`` (124M), ``gpt2-medium``
  (355M), reported as reference numbers, not training-matched.

## 5. Results

> Numbers go here once the ablation suite (5 seeds × 7 configs) has
> run on the actual GPU/machine. Tables should include:
> - **Table 1**: matched-compute comparison vs Vanilla-Transformer
>   on fineweb-edu val PPL and downstream accuracy.
> - **Table 2**: ablation — full UCSA vs UCSA without JEPA chain,
>   without EMA, without input-reconstruction, without
>   text-conditioner.
> - **Table 3**: scaling — UCSA at 63M, 130M, 350M params
>   vs vanilla-Transformer at the same three sizes.
> - **Table 4**: memory-bank probing — what does each bank learn?
>   Retention-score distribution snapshots.

## 6. Discussion

If the matched-compute comparison holds, the contribution is not
that UCSA beats a 50× larger model — it doesn't have to — but that
the PCS-bounded design plus the multi-step JEPA chain yields a
small architecture that's competitive with a vanilla Transformer
of comparable size, with a different inductive bias (state-centric
vs sequence-centric). The build-vs-buy question then moves to
infrastructure-bounded deployments: keep one state across many
calls, swap the operator.

## 7. Conclusions

UCSA demonstrates that a cognitive architecture centred on a
single persistent state, paired with a multi-step JEPA prediction
chain over the reasoning loop, is a viable design point for small
foundation models. The matched-compute evaluation, ablations, and
standard-LM benchmarks are open-sourced to make the claims
reproducible.

## Appendix A. Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
.venv/bin/python scripts/train.py             # UCSA-small @ 12k steps
.venv/bin/python scripts/train_baseline.py  # vanilla-Transformer @ 12k steps
.venv/bin/python scripts/eval.py \
    --ucsa-ckpt ckpts/ucsa-final.safetensors \
    --baseline-results runs/baseline.json \
    --out-json runs/eval.json
pytest -q
ruff check ucsa scripts
```

Random seeds default to ``42``. ``--seed N`` overrides. Full JSON
artifacts land under ``runs/``.

## Appendix B. Hyperparameters

See ``ucsa/configs/default.yaml`` for the full list and
``scripts/train.py`` defaults.

## Appendix C. PCS Retention Score

Retention score combines importance, usage, recency:

$$
R(t) = \alpha I(t) + \beta U(t) - \gamma A(t)
$$

with default weights ``alpha=0.5``, ``beta=0.3``, ``gamma=0.2`` and
a ``0.01`` floor (tokens below the floor are recycled on the next
write).

## Appendix D. Bibliography (selected)

- Assran et al., I-JEPA, 2023.
- Maes et al., LeWorldModel, arXiv 2603.19312, 2026.
- Huang et al., TC-JEPA, arXiv 2605.03245, 2026.
- Bowne-Anderson / Raschka, "LLM Architecture in 2026", 2026.
- Raschka, "Recent Developments in LLM Architectures", 2026.
- DeepSeek-AI et al., DeepSeek-V2 / V3 / V4 papers.
- Moonshot AI, Kimi K3, 2026.
- Keller Jordan, Muon optimiser, 2024.
