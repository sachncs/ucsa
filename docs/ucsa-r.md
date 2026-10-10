# UCSA-R: causal, windowed, state-carrying language model

UCSA-R keeps UCSA's central idea, one persistent multi-bank state that
everything reads and writes, and makes the information flow causal by
construction. Every logit is an exact next-token prediction, so the model is
directly comparable with any causal language model and is also a lossless
compressor.

Code: `ucsa/models/recurrent.py` (model), `ucsa/training/engine.py` (training),
`ucsa/training/shards.py` (data), `ucsa/arithmetic.py` (compression).

## Why it exists

The original UCSA decodes its logits from working-bank slots that have read
the whole input, so a slot can copy the token it is scored on. Measured on a
trained checkpoint: validation perplexity 220 with the answers visible in the
input and 13,611 with them hidden. UCSA-R removes that route instead of
working around it.

## Data flow

A sequence is cut into chunks of `chunk_size` tokens (128 by default).

```
tokens --embed--> x_t --chunk encoder (bidirectional, per chunk)--> e_t
S_t     = update(S_{t-1}, e_t)                  # tiny sequential scan
logits_t = decode(x_t | S_{t-1}, window)        # all chunks in parallel
```

* **Decoder.** Each token attends over `[state slots | previous chunk |
  current chunk]` with one sliding-window attention. A token sees exactly the
  last `window` tokens (default: one chunk) plus the state, and the mask is
  causal inside the current chunk. The previous chunk's keys and values are
  the same layer's keys and values shifted by one chunk, so the window costs
  no extra projection.
* **State.** `S` is `num_slots x hidden`, split into banks (working,
  long-term, intent). It is written once per chunk by a gated, optionally
  sparse update; per-slot biases give each bank its own retention. Chunk `t`
  reads `S_{t-1}` only.
* **Parallel and cheap.** The state is built from shallow chunk summaries, so
  the only sequential work is a small scan. The expensive decoder runs over
  all chunks as one batch.
* **Constant memory.** The state has a fixed size, so the same weights run
  over arbitrarily long streams; generation keeps only the state, one chunk's
  cache and the current partial chunk.

## What can be switched

All fields of `recurrent.Config` are validated; unknown keys are rejected.
The ones that define the research questions:

| Field | Meaning |
|---|---|
| `use_state` | `False` resets the state every chunk: the chunk-local control. |
| `window` | Sliding window in tokens. `None` is one chunk; `0` is chunk-local attention. |
| `read_gate` | Slot keys and values are scaled by a learned scalar that starts at 0, so the model begins unable to use the state. |
| `surprise_gate` | The JEPA predictor's error on a chunk raises the write gate: store what prediction left unexplained. |
| `slot_dropout`, `min_slots` | Train on random prefixes of the slots, so memory can be shrunk at inference. |
| `write_top_k` | Only the top-k slots are updated per chunk. |
| `jepa_weight` | Weight of the causal JEPA loss (predict a chunk latent from the previous state). |

## Guarantees, and the tests that hold them

| Guarantee | Test |
|---|---|
| Strictly causal: changing token `p` never changes a logit before `p`. | `tests/test_recurrent.py` |
| Exact window: with one layer, chunk-1 position `i` depends on chunk-0 position `j` iff `j > i`. | `tests/test_recurrent.py` |
| Streaming equals teacher forcing at every position, across chunk boundaries. | `tests/test_recurrent.py` |
| Lossless: arithmetic coding with the model round-trips exactly; stored size equals the cross-entropy; a model that peeks at the future cannot be decoded. | `tests/test_arithmetic.py` |
| One float type everywhere; no float-to-float cast, autocast or loss scaler in the library. | `tests/test_single_precision.py` |
| Training carries no instrumentation; auditing lives in a dry run. | `tests/test_single_precision.py`, `tests/test_dryrun.py` |

## Workflow

```bash
python scripts/prepare_data.py --out data          # tokenise once, dedupe
python scripts/dry_run.py --preset small           # dtype audit, memory, speed
python scripts/train_r.py --preset small --out-dir ckpts/r-small
python scripts/eval.py --recurrent-ckpt ckpts/r-small/final.pt
python scripts/compress.py --ckpt ckpts/r-small/final.pt   # real file sizes
python scripts/probe_state.py --ckpt ckpts/r-small/final.pt \
    --control ckpts/r-nostate/final.pt             # what the state buys
python scripts/probe_reset.py --ckpt ckpts/r-small/final.pt  # is its content used?
```

Choosing a design that holds at full length (not just after 600 steps):

```bash
python scripts/ablate.py --steps 600 --out runs/ablate-window   # filter
python scripts/ladder.py --arms base no-state --out runs/ladder  # 150..1200
python scripts/ladder.py --analyze --target 12000 --out runs/ladder
```

`ladder.py` fits `L(t) = floor + A t^-alpha` to each arm, forecasts the loss
at the full-run length with an interval driven by the measured seed noise,
flags curves that cross before then, and only calls an arm better or worse
when the forecast gap's interval excludes zero.

## Evaluation

* **Bits per byte** is reported next to perplexity. It does not depend on the
  tokenizer and is set against zlib and LZMA on the same bytes.
* **Benchmarks** (HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande) use
  the lm-evaluation-harness prompts and metrics (`acc_norm` for HellaSwag and
  ARC-Challenge, `acc` elsewhere, partial scoring for WinoGrande) on the full
  splits, and are compared with published numbers in
  `paper/reference_results.json`.
* **State probes** show what the state contributes: loss by chunk position,
  windows longer than any seen in training, and a bits-per-byte versus
  state-slots curve.

## Precision

The library uses one floating-point type, `ucsa.utils.precision.DTYPE`
(float32; `UCSA_DTYPE=float16` switches the whole library). Integer and
boolean tensors are converted exactly with `precision.to_dtype`. In float16,
PyTorch's own fused kernels (RMSNorm, attention) upcast internally; those are
reported by the dry run, not hidden.
