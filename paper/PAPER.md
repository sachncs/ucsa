# UCSA-R: Auditing and Rebuilding a Persistent-State Language Model

## Abstract

UCSA routes all computation through one persistent, multi-bank state. We
audited it and found that its language head could read the tokens it was scored
on: held-out perplexity was 220 with targets visible and 13,611 with them
hidden. We rebuilt it as UCSA-R, a causal chunked model whose logits are exact
next-token predictions, and evaluate it by compression (bits per byte, with a
lossless arithmetic coder as a causality proof) and on the full splits of five
benchmarks beside published numbers. A 19M-parameter UCSA-R trained for 12,000
steps (98M tokens, 2.8 hours on one laptop GPU) reaches 1.39 bits per byte on
held-out FineWeb-Edu and compresses text to 1.48 bits per byte, against 3.5 for
LZMA. Benchmark accuracy is near chance and below published 130M-160M models
trained on 300B tokens. Our central negative result: against a matched model
with no persistent state, trained identically for 12,000 steps, the state gives
no measurable gain (+0.008 bits/token, inside a 0.015 noise floor). The
contributions that hold are the leak audit, the windowed causal design, the
evaluation and model-selection methodology, and the engineering.

## 1. Introduction

A language model assigns a probability to every next token, and a probability
is a code length. Perplexity is bits per token; with the tokenizer divided out
it is bits per byte, the size of the text under the model's own compressor.
This paper treats that equivalence as a working tool, not a slogan: it gives us
a tokenizer-independent metric that stays informative when benchmark accuracy
sits at chance, a lossless compressor whose round trip is an operational test
of causality, and a way to ask what a persistent memory is worth in bits.

UCSA's original architecture routes every computation through one persistent,
multi-bank cognitive state. Auditing it for this work turned up a flaw that
invalidates its earlier evaluation. We report that flaw, fix it by redesign,
and then do the work properly: a causal windowed model that keeps the state,
evaluation against published numbers on the full benchmark splits, and a
statistical procedure for choosing designs that survives the jump from a
short experiment to the full run.

**Contributions.**

1. **A measured defect in the original model.** Its language head reads the
   same sequence it is scored on and can copy the target token: held-out
   perplexity was 220 with the targets visible and 13,611 with them hidden
   (Section 2).
2. **UCSA-R**, a causal language model with a persistent multi-bank state and
   sliding-window attention over `[state | previous chunk | current chunk]`.
   Every logit is an exact next-token prediction. Replacing cross-attention by
   the window removed 7.6% of the parameters and lowered perplexity from 576 to
   463 at equal speed (Section 3).
3. **Compression as verification.** A lossless arithmetic coder driven by the
   model round-trips exactly across chunk boundaries; a model that peeks at
   the future compresses spectacularly and cannot be decoded. Bits per byte,
   with zlib and LZMA as anchors, replaces perplexity as the headline metric
   (Section 4.2).
4. **A procedure that carries short experiments to the full run**: paired
   statistics on identical windows, a measured run-to-run noise floor, and
   learning-curve forecasts with crossover detection (Section 4.3). It showed
   that an ungated state read is harmful early in training, and that a
   zero-initialised gate removes that cost; the full run then found the state
   itself gives no measurable gain (Section 5).
5. **Engineering that makes the numbers trustworthy**: one float type enforced
   by a static and a runtime audit, an auditor that lives in a dry run and never
   in training, atomic data shards, resumable training with a single host
   synchronisation per step, and tests organised by guarantee (Section 6).

## 2. A flaw in the original architecture

The original UCSA writes its whole input into a cognitive state and decodes the
logits from 64 working-bank slots that have read that state. The language-model
target for slot *i* is a token that is also in the input, so the slot can
learn to copy it. On a checkpoint trained for 1,500 steps:

| input to the model | held-out perplexity |
|---|---|
| target tokens visible | 220 |
| target tokens removed from the input | 13,611 |

The first number looks like a strong language model; the second is near the
unigram level. The two evaluation failures that surrounded it (a script that
scored a randomly initialised model because its configuration disagreed with
the checkpoint, and a benchmark scorer that placed the answer choice in the
model's input) would each have produced optimistic numbers on their own. All
three are fixed and covered by regression tests (see the changelog). Nothing
in the original paper's language-modelling or benchmark numbers should be
relied on.

## 3. Method

### 3.1 UCSA-R

A sequence is cut into chunks of 128 tokens. A shallow bidirectional encoder
summarises each chunk, and a gated update writes the summary into a state of
32 slots (working, long-term and intent banks with their own retention
biases). The decoder runs over all chunks as one batch; each token attends over
the state, the previous chunk and the current chunk with one sliding-window
attention, causal inside the current chunk. Chunk *t* reads the state written
from chunks before *t* only. The state has constant size, so the same weights
run over streams of any length, and generation keeps only the state, one
chunk's cache and the current partial chunk.

### 3.2 Why a sliding window replaces cross-attention

The first version read the state with cross-attention in every block and
decoded each chunk with no view of the previous one. Replacing both
cross-attention uses by sliding-window attention, with the state slots as
always-visible memory tokens, removed the cross-attention weights (19.16M
parameters against 20.73M), kept the speed (9.7k against 9.9k tokens/s), lowered
perplexity at 600 steps from 576.2 to 463.2, and cut run-to-run variation about
four-fold. Removing the window from the new design raises the loss by 0.095
bits/token (perplexity 495), about five times the noise, so the window itself
is responsible, not an incidental change.

### 3.3 A zero-initialised read gate

With the state read ungated, the stateful model was *worse* than the same model
without a state (by 0.054 bits/token at 600 steps). Scaling the keys and values
of the state slots in each block by a learned scalar that starts at zero makes
the model begin unable to use the state, so it starts equal to the control and
opens the gate only as far as that helps. At zero, every slot logit is a
constant and every value vanishes, which is tested exactly. The gated model
beats the ungated one by 0.06 to 0.09 bits/token across replicate runs.

### 3.4 What did not help

Measured on the gated base at 600 steps against a noise floor of about
0.015 bits/token per run (Section 4.3):

* The causal JEPA objective and the surprise-gated write (JEPA's prediction
  error raising the write gate) were inside the noise (-0.027 and -0.012). The
  JEPA target network was also found, in review, to track an online encoder that
  receives no gradient, so its EMA was inert. JEPA is off in the final model.
* Weight averaging made the model worse (+0.035; it lags a model that is still
  improving fast). Filtering training documents by zlib compressibility was
  neutral (-0.022).
* The state itself was neutral at 600 steps (443.8 without it, 444.0 with it).

## 4. Evaluation

### 4.1 Benchmarks

HellaSwag, ARC-Easy, ARC-Challenge, PIQA and WinoGrande are scored as in
EleutherAI's lm-evaluation-harness: the same prompts, `acc_norm` for HellaSwag
and ARC-Challenge, `acc` elsewhere, partial scoring for WinoGrande, and the
full evaluation splits. A slot model never sees the answer; a causal model is
scored on the same tokens. Results are set beside published zero-shot numbers
for Pythia-160M, Hybrid H3-130M and Mamba-130M (Gu and Dao, 2023, Table 3), and
beside chance. Those models saw 300 billion tokens; ours sees about 100
million, so we report the ratio of data used and do not claim parity.

### 4.2 Compression

Bits per byte divides the model's code length by the raw bytes of the text. We
report it with zlib and LZMA on the same bytes. The arithmetic coder is the
integer range coder of Witten, Neal and Cleary. Its round trip is exact, the
stored size equals the teacher-forced cross-entropy, and a decoder given only
the past cannot reproduce a model that used the future. A compressed file
decodes only with the software and device that wrote it, because model
probabilities are floating point.

### 4.3 Choosing designs that hold at full length

A 600-step comparison can mislead for a 12,000-step run: rankings can flip,
and the best learning rate usually shrinks with training length. We use three
tools.

1. **Paired comparison on identical windows**, with a bootstrap interval, which
   removes window difficulty but measures only evaluation noise.
2. **A run-to-run noise floor.** Repeating the same configuration gave
   perplexities of 436.0, 444.0 and 444.3: training on this GPU is not bitwise
   deterministic, and the standard deviation of a run is about 0.015
   bits/token. A gap must exceed twice that to be called real.
3. **Learning-curve forecasts.** Each candidate is trained for 150, 300, 600
   and 1,200 steps, each with its own annealed schedule; `L(t) = floor +
   A t^-alpha` is fitted and extrapolated to 12,000 steps with an interval
   driven by the noise floor. Curves that cross before 12,000 steps are
   flagged, and a candidate is called better or worse only when the interval of
   the forecast gap excludes zero.

## 5. Results

All numbers are generated into `paper/RESULTS.md` by `scripts/report.py` from
the files in `paper/artifacts/`.

**Language modelling.** The final model has 19.16M parameters, width 256, six
layers, chunk 128, and 32 state slots. After 12,000 steps (batch 8 x 1,024, 98M
tokens, 2.79 h) held-out perplexity is 82.3 on FineWeb-Edu (1.390 bits per
byte) and 244 on WikiText-103 (out of domain). The loss falls monotonically
through training (perplexity 247.7, 140.3, 102.0, 86.6, 82.3 at steps 1k, 3k,
7k, 10k, 12k).

**Compression.** With arithmetic coding the model stores 4,096 tokens (18,436
bytes) in 3,417 bytes: 1.483 bits per byte, round trip verified lossless, against
3.611 for zlib and 3.508 for LZMA. On 16,384 tokens of other text it reaches
1.526 against 3.175 and 2.964.

**Benchmarks.** Zero-shot, full splits (PIQA is a 1,000-example subset), against
published numbers (Gu and Dao, 2023, Table 3):

| task | metric | ours | chance | Pythia-160M | H3-130M | Mamba-130M |
|---|---|---|---|---|---|---|
| HellaSwag | acc_norm | 26.1 ± 0.4 | 25.0 | 30.2 | 31.7 | 35.3 |
| PIQA | acc | 54.8 ± 1.6 | 50.0 | 61.4 | 64.2 | 64.5 |
| ARC-Easy | acc | 34.3 ± 1.0 | 25.0 | 43.2 | 44.4 | 48.0 |
| ARC-Challenge | acc_norm | 21.2 ± 1.2 | 25.0 | 24.1 | 24.2 | 24.3 |
| WinoGrande | acc | 49.1 ± 1.4 | 50.0 | 51.9 | 50.6 | 51.9 |

The model is above chance on HellaSwag, PIQA and ARC-Easy, at or below chance
on ARC-Challenge and WinoGrande, and below every published model on every task.
With 3,000 times fewer training tokens this is expected, and we do not claim
competitiveness. ARC-Challenge, 3.8 points below chance (3 standard errors), is
the one result we cannot explain.

**Does the persistent state help?** This is the question the architecture
exists to answer. We trained an identical model with the state reset every chunk
(same data, schedule and seed). After 12,000 steps:

| run | perplexity | bits per byte |
|---|---|---|
| with state | 82.33 | 1.3902 |
| no state | 81.89 | 1.3885 |

The gap, +0.0078 bits/token in the state's disfavour, is inside the 0.015
noise floor; with one seed per arm we cannot say the state is harmful, only that
it is not measurably helpful. The per-chunk comparison agrees: on 1,024-token
and 4,096-token windows the state-minus-control loss is within +-0.05
bits/token at every chunk position, with no growth along the window, so the
state does not accumulate useful information over distance at this scale. Reading
fewer slots does hurt (1.355 bits per byte with 32 slots, 1.375 with 16, 1.453
with 8, 2.032 with 1), yet the stateless model is as good. A direct test
explains this: resetting the trained model's state to its initial value at every
chunk, on the same 200 windows, raises the loss by only 0.014 bits/token (95%
interval 0.013 to 0.015), 0.2% of the loss. What the state remembers from
earlier chunks is almost unused; the slots work as extra learned context tokens
that every chunk reads, not as memory of the past.

A recall probe agrees. We plant a random 32-token span, add real text, and repeat
the span; with a gap above one chunk only the state could know it. The repeat
costs the same as a fresh random span (18.87 against 18.90 bits per token at a
gap of 512, and 18.90 against 18.91 for the control), while inside the window it
is cheap (11.2 against 17.5 bits). Trained on text, the state stores nothing
that survives past the attention window.

**Design choices at 600 steps** (noise floor 0.0145 bits/token; Section 4.3):

| change | effect (bits/token) | verdict |
|---|---|---|
| sliding window replaces cross-attention | perplexity 576 to 463, -7.6% params | better |
| remove the window | +0.095 | worse |
| zero-initialised read gate | -0.087 | better |
| learning rate 2.4e-3 against 6e-4 | -0.339 | better |
| learning rate 4.8e-3 | +0.131 | worse |
| JEPA loss off | -0.027 | within noise |
| surprise-gated write | -0.012 | within noise |
| weight averaging | +0.035 | worse |
| compressibility filter | -0.022 | within noise |
| no state | -0.001 | no difference |

The 12,000-step result confirms the state row and the learning-rate choice (the
final run used 2.4e-3). It does not test the other rows, which were not rerun at
full length.

**The forecast failed.** From rungs of 150 to 1,200 steps we forecast
perplexity 47.7 [42.9, 54.8] at 12,000 steps for the chosen learning rate,
recorded before the run. The outcome was 82.3, 0.79 bits/token above the
interval. The power-law fit extrapolated ten times beyond its data and was
over-optimistic. The learning-rate ranking it supported did hold, but the
absolute forecast should not be trusted, and the intervals we derived from
run-to-run noise understate extrapolation error.

## 6. Engineering

The measurements above are only as good as the code that produces them, so the
engineering is part of the method. Each item below was added because a concrete
failure was found, and each has a regression test.

* **One float type.** The library uses a single floating-point dtype and no
  autocast, loss scaler or float-to-float cast. A static test rejects any such
  construct in every source file, and a runtime auditor records the dtype of
  every tensor any operator produces during a real training step. The auditor
  allocates memory and slows execution, so it lives in a dry run
  (`scripts/dry_run.py`) that gates training and is never imported by it; a test
  enforces that. In float16 the audit reports float32 results from inside
  PyTorch's own fused kernels (RMSNorm, attention); these are surfaced, not
  hidden.
* **Data.** Corpora are tokenised once into memory-mapped shards. A batch is a
  pure function of `(seed, step)`, so training resumes on exactly the data
  stream it left, and training never touches the network (a stalled stream
  hung earlier runs). Shards are written atomically; an in-place rewrite once
  truncated a file under a reader that had it mapped. Documents are
  deduplicated by their first 128 tokens.
* **Training.** Resumable from the latest checkpoint, one host
  synchronisation per step, a single concatenated gradient-norm (the per-tensor
  version cost 10% of a step on this backend), a skip-on-non-finite guard, no
  weight decay on biases and gates, and full validation of every setting.
* **Tests by guarantee, not by method.** Strict causality, the exact window
  boundary, streaming equal to teacher forcing, lossless round trips, and a
  fuzz test over random configurations. Reviewing the suite this way found
  defects that example-based tests had not: a curator that silently dropped
  work after a restart, a pruning routine that recycled empty slots, a
  verifier that accepted an empty or NaN candidate, a configuration check that
  crashed on `heads=0`, and a default that rejected valid small layouts.

## 7. Limitations

* **Scale.** The model has 19M parameters and sees about 100M tokens. Published
  comparison models saw 300B. On the benchmarks, small models are near chance
  and the standard error of a 1,000-example task is about 1.5 points, so most
  benchmark differences are not resolvable here. Bits per byte is the more
  informative number at this scale.
* **The persistent state.** At 12,000 steps the state gives no measurable gain
  over a matched stateless model (Section 5), with one seed per arm. Training
  windows are 1,024 tokens; a constant-size state could matter at much longer
  contexts or larger scale, which we did not test.
* **Noise.** Training is not bitwise reproducible on this hardware. The noise
  floor was measured from repeated same-configuration runs and is about 0.015
  bits/token per run. Conclusions near that size are not claimed.
* **Extrapolation.** Forecasts reach 10x beyond the longest ladder run. The
  forecast for the chosen learning rate missed the 12,000-step outcome
  (Section 5).
* **Compression.** A compressed file decodes only on the software and device
  that wrote it, because model probabilities are floating point.
* **Not explored.** Chunk size, depth and width beyond one measured sweep,
  longer training sequences, and larger scale.

## 8. Reproduction

```bash
scripts/reproduce.sh              # data, dry run, training, evaluation, report
```

Every number in Section 5 is generated into `paper/RESULTS.md` by
`scripts/report.py` from run artifacts; the experiments, not the document, are
the source of truth. The code, ablation records and the exact commit used for
the final run are in the repository.
