export const SITE_CONFIG = {
  repo: 'sachncs/ucsa',
  repoUrl: 'https://github.com/sachncs/ucsa',
  docsUrl: 'https://github.com/sachncs/ucsa/blob/master/docs',
  paperUrl: 'https://github.com/sachncs/ucsa/blob/master/paper/PAPER.md',
  paperTablesUrl: 'https://github.com/sachncs/ucsa/blob/master/paper/TABLES.md',
  issuesUrl: 'https://github.com/sachncs/ucsa/issues',
  discussionsUrl: 'https://github.com/sachncs/ucsa/discussions',
  contributingUrl: 'https://github.com/sachncs/ucsa/blob/master/CONTRIBUTING.md',
  changelogUrl: 'https://github.com/sachncs/ucsa/blob/master/CHANGELOG.md',
  release: 'v0.1',
  license: 'Apache-2.0',
} as const;

export const METRICS = [
  {
    icon: 'check',
    value: '607',
    suffix: 'tests',
    title: 'Test suite',
    desc:
      'Every subsystem ships with tests — PCS, operator, reasoning loop, projection heads, JEPA chain, EMA, curriculum, trainer, eval harness, intent descent.',
  },
  {
    icon: 'shield',
    value: '80',
    suffix: '%',
    title: 'Coverage gate',
    desc:
      'CI fails on any coverage drop below 80%. Measured per branch and uploaded to Codecov on every push.',
  },
  {
    icon: 'python',
    value: '3.11 · 3.12',
    suffix: '',
    title: 'Python support',
    desc:
      'CI matrix runs on Python 3.11 and 3.12. PyTorch 2.1+, MPS, CUDA, and CPU are all in the supported matrix.',
  },
  {
    icon: 'license',
    value: 'Apache',
    suffix: '-2.0',
    title: 'License',
    desc:
      'Apache-2.0 from day one. The full LICENSE file is at the repo root and matches the declaration in pyproject.toml.',
  },
  {
    icon: 'doc',
    value: 'Paper',
    suffix: 'draft',
    title: 'Working paper',
    desc:
      'A full paper draft lives in paper/PAPER.md. It records what the system measures, what is in the repo, and what is left to run at paper-grade scale.',
  },
  {
    icon: 'compare',
    value: 'matched',
    suffix: 'compute',
    title: 'Matched-compute baselines',
    desc:
      'Every UCSA result is paired with a vanilla-Transformer baseline of identical parameter count, dataset, and step budget. The protocol catches negative results.',
  },
  {
    icon: 'beaker',
    value: '5',
    suffix: 'tasks',
    title: 'Standard eval harness',
    desc:
      'HellaSwag, ARC-easy, ARC-challenge, PIQA, and WinoGrande via rank-by-log-likelihood. Deterministic by seed.',
  },
  {
    icon: 'config',
    value: 'Hydra',
    suffix: '/OmegaConf',
    title: 'Reproducible configs',
    desc:
      'Every run is reproducible from a config dump. CLI overrides are recorded in the JSON output.',
  },
] as const;

export const BANKS: ReadonlyArray<{
  name: string;
  tokens: string;
  role: string;
  color: string;
  full?: boolean;
  meta: {
    role: string;
    path: string;
    tokens: string;
    service?: string;
  };
}> = [
  {
    name: 'working',
    tokens: '64 tok',
    role: 'Scratch space mutated by every reasoning step. Every head reads from this bank.',
    color: 'var(--bank-working)',
    meta: { role: 'scratch', path: 'in stream', tokens: 'shape (64, hidden)' },
  },
  {
    name: 'long_term',
    tokens: '128 tok',
    role: 'Accepted knowledge, retained across requests. Each token carries retention metadata that drives the recycle policy.',
    color: 'var(--bank-long-term)',
    meta: { role: 'memory', path: 'in stream', tokens: 'shape (128, hidden)' },
  },
  {
    name: 'goal',
    tokens: '16 tok',
    role: 'Holds the active objective. Mutated when the active goal changes; otherwise a stable anchor for the reasoning loop.',
    color: 'var(--bank-goal)',
    meta: { role: 'objective', path: 'in stream', tokens: 'shape (16, hidden)' },
  },
  {
    name: 'episode',
    tokens: '32 tok',
    role: 'Per-request context buffer. Holds the immediate working memory between the start and end of a single request.',
    color: 'var(--bank-episode)',
    meta: { role: 'context', path: 'in stream', tokens: 'shape (32, hidden)' },
  },
  {
    name: 'task',
    tokens: '16 tok',
    role: 'Long-running task state. Persists across episodes when a multi-request task is in flight.',
    color: 'var(--bank-task)',
    meta: { role: 'task state', path: 'in stream', tokens: 'shape (16, hidden)' },
  },
  {
    name: 'memory_index',
    tokens: '32 tok',
    role: 'Retrieval index, cross-attended by every transformer block. Holds the keys and values for retrieval.',
    color: 'var(--bank-memory-index)',
    meta: { role: 'retrieval', path: 'cross-attn', tokens: 'shape (32, hidden)' },
  },
  {
    name: 'intent',
    tokens: '16 tok',
    role: 'Origination signal. Held out of the operator stream by design — the OriginationHead reads intent and working memory and produces the next iteration’s input. Per-slot attribution is well-posed because there is one path, not many.',
    color: 'var(--bank-intent)',
    full: true,
    meta: {
      role: 'origination',
      path: 'held out · origination',
      tokens: 'shape (16, hidden)',
      service: 'OriginationHead + IntentUpdate',
    },
  },
];

export const ARCH_CARDS = [
  {
    n: '01',
    name: 'PCS — Persistent Cognitive State',
    desc: 'Seven differentiable token banks. Every read and write in the system is a slice of this state.',
    tag: 'ucsa/models/state.py',
    span: 'wide' as const,
  },
  {
    n: '02',
    name: 'Operator — state transition F',
    desc: 'The only computation engine. Maps (Cₜ, Oₜ) → Cₜ₊₁. Reference impl is a pre-norm Transformer with GQA, MoE, and memory-index cross-attention.',
    tag: 'ucsa/models/transformer_operator.py',
    span: 'wide' as const,
  },
  {
    n: '03',
    name: 'Reasoning loop',
    desc: 'Runs F N times per forward. Carries differentiable bank tensors between iterations so the loss can reach the operator.',
    tag: 'reasoning_loop.py',
    span: 'half' as const,
  },
  {
    n: '04',
    name: 'Memory service',
    desc: 'Background worker. Verifies, consolidates, and prunes long-term memory. Inference never blocks on memory.',
    tag: 'memory_service.py',
    span: 'half' as const,
  },
  {
    n: '05',
    name: 'Projection heads',
    desc: 'Four heads — language, planning, tool, memory — plus the input-reconstruction and origination heads.',
    tag: 'projection_heads.py',
    span: 'half' as const,
  },
  {
    n: '06',
    name: 'EMA target encoder',
    desc: 'Frozen EMA copy of the model. Provides the JEPA chain’s targets so the prediction chain stays stable.',
    tag: 'training/ema.py',
    span: 'half' as const,
  },
  {
    n: '07',
    name: 'Auxiliary losses',
    desc: 'JEPA chain, input-reconstruction capacity bottleneck, memory stability, and MoE load-balancing.',
    tag: 'models/losses.py',
    span: 'half' as const,
  },
  {
    n: '08',
    name: 'Eval harness',
    desc: 'Seed-deterministic rank-by-log-likelihood on HellaSwag, ARC, PIQA, and WinoGrande.',
    tag: 'training/eval_harness.py',
    span: 'half' as const,
  },
] as const;

export const TIMELINE_STEPS = [
  {
    title: 'Input arrives',
    desc: 'A tokenised sequence lands in the trainer. Targets are the next-token ids with ignore_index on padding.',
  },
  {
    title: 'Perception & routing',
    desc: 'Tokens are embedded and projected to the operator’s hidden size. MoE router logits are computed if MoE is on.',
  },
  {
    title: 'PCS banks update',
    desc: 'The observation is injected into the working bank. Long-term, goal, episode, task, and memory-index banks are concatenated into the operator stream.',
  },
  {
    title: 'Reasoning loop iterates',
    desc: 'F runs N=4 times. Each iteration carries the previous iteration’s differentiable banks so the loss can reach the operator.',
  },
  {
    title: 'Memory service syncs',
    desc: 'A background worker verifies, consolidates, and prunes long-term memory. Inference never blocks.',
  },
  {
    title: 'Heads read the state',
    desc: 'Language, planning, tool, and memory heads project working memory to their respective outputs.',
  },
  {
    title: 'JEPA chain predicts',
    desc: 'Consecutive working latents are paired into (predicted, target) tuples. Targets come from the EMA encoder for stability.',
  },
  {
    title: 'Outputs are generated',
    desc: 'The combined loss flows back through every projection that touched the PCS — including the operator.',
  },
] as const;

export const ROLES: ReadonlyArray<{
  code: string;
  title: string;
  desc: string;
  links: ReadonlyArray<{ label: string; href: string; external?: boolean }>;
}> = [
  {
    code: 'IM',
    title: 'Implementers',
    desc: 'Build UCSA, swap the operator, add a bank, wire a custom loss. You want the API tour and the tutorials.',
    links: [
      { label: 'API reference', href: `${SITE_CONFIG.docsUrl}/api-reference.md` },
      { label: 'Tutorials', href: `${SITE_CONFIG.docsUrl}/tutorials.md` },
    ],
  },
  {
    code: 'RS',
    title: 'Researchers',
    desc: 'Understand the thesis, follow the math, scrutinise the matched-compute protocol. You want the architecture and the paper.',
    links: [
      { label: 'Architecture', href: `${SITE_CONFIG.docsUrl}/architecture.md` },
      { label: 'Paper draft', href: SITE_CONFIG.paperUrl },
    ],
  },
  {
    code: 'EV',
    title: 'Evaluators',
    desc: 'Rerun the numbers. The harness is deterministic and the protocol catches negative results.',
    links: [
      { label: 'Paper tables', href: SITE_CONFIG.paperTablesUrl },
      { label: 'Get started', href: `${SITE_CONFIG.docsUrl}/getting-started.md` },
    ],
  },
  {
    code: 'CT',
    title: 'Contributors',
    desc: 'Open a PR. The dev setup is short, the test suite is fast, and the issue templates are short.',
    links: [
      { label: 'Contributing', href: SITE_CONFIG.contributingUrl },
      { label: 'Issues ↗', href: SITE_CONFIG.issuesUrl, external: true },
    ],
  },
];

export const DOCS = [
  {
    eyebrow: 'onboarding',
    title: 'Getting started',
    desc: 'Install, smoke test, full reproduction, ablation flags.',
    href: `${SITE_CONFIG.docsUrl}/getting-started.md`,
  },
  {
    eyebrow: 'design',
    title: 'Architecture',
    desc: 'Deep design notes for every subsystem, with the maths.',
    href: `${SITE_CONFIG.docsUrl}/architecture.md`,
  },
  {
    eyebrow: 'reference',
    title: 'API reference',
    desc: 'Module-by-module tour of ucsa/, organised by responsibility.',
    href: `${SITE_CONFIG.docsUrl}/api-reference.md`,
  },
  {
    eyebrow: 'walkthroughs',
    title: 'Tutorials',
    desc: 'Five end-to-end examples: build, customise, ablate, measure, probe.',
    href: `${SITE_CONFIG.docsUrl}/tutorials.md`,
  },
  {
    eyebrow: 'dev workflow',
    title: 'Contributing',
    desc: 'Setup, lint, type-check, tests, PR flow, code review checklist.',
    href: SITE_CONFIG.contributingUrl,
  },
  {
    eyebrow: 'release notes',
    title: 'Changelog',
    desc: 'Per-phase history of the project, with the matching tests added.',
    href: SITE_CONFIG.changelogUrl,
  },
  {
    eyebrow: 'research',
    title: 'Paper draft',
    desc: 'The full paper draft, including the negative-results section.',
    href: SITE_CONFIG.paperUrl,
  },
  {
    eyebrow: 'results',
    title: 'Paper tables',
    desc: 'Generated from runs/*.json. The numbers the paper claims.',
    href: SITE_CONFIG.paperTablesUrl,
  },
  {
    eyebrow: 'help',
    title: 'Discussions',
    desc: 'Where to ask design questions and propose changes.',
    href: SITE_CONFIG.discussionsUrl,
  },
] as const;

export const PAPER_ROWS = [
  {
    name: 'vanilla-transformer',
    params: '63M',
    ppl: '1.0413 ± 0.0053',
    delta: '—',
  },
  {
    name: 'ucsa-full',
    params: '63M',
    ppl: '1.0610 ± 0.0119',
    delta: '+0.0197',
  },
  {
    name: 'ucsa-no-jepa',
    params: '63M',
    ppl: '—',
    delta: 'ablation',
  },
  {
    name: 'ucsa-no-ema',
    params: '63M',
    ppl: '—',
    delta: 'ablation',
  },
  {
    name: 'ucsa-no-recon',
    params: '63M',
    ppl: '—',
    delta: 'ablation',
  },
] as const;

export const CONTRIB_STEPS = [
  {
    num: '01',
    title: 'Dev setup',
    desc: 'Clone, create a venv, install with the [dev] extra.',
    cmd: 'pip install -e ".[dev]"',
  },
  {
    num: '02',
    title: 'Lint & format',
    desc: 'Ruff for lint, black for format. Runs in CI; failures block the PR.',
    cmd: 'ruff check ucsa tests scripts',
  },
  {
    num: '03',
    title: 'Type-check',
    desc: 'Mypy on ucsa/. CI on Python 3.11 and 3.12.',
    cmd: 'mypy ucsa',
  },
  {
    num: '04',
    title: 'Tests',
    desc: '600 fast tests (607 total). Slow marker runs in a separate job. New behaviour must ship with a test.',
    cmd: 'pytest -q -m "not slow"',
  },
  {
    num: '05',
    title: 'Smoke test',
    desc: 'Five training steps. Exits in under a minute on CPU. Catches wiring regressions.',
    cmd: 'scripts/train.py --max-steps 5 --skip-baselines',
  },
  {
    num: '06',
    title: 'PR & review',
    desc: 'Conventional Commits, filled-in PR template, code review checklist in the contributor guide.',
    cmd: 'git commit -m "feat: ..."',
  },
] as const;

export const STRIP = [
  { label: 'tests', value: '607 passing' },
  { label: 'coverage', value: '80% enforced' },
  { label: 'python', value: '3.11 · 3.12' },
  { label: 'pytorch', value: '2.1+ · MPS · CUDA' },
  { label: 'license', value: 'Apache-2.0' },
  { label: 'release', value: 'v0.1 · research' },
] as const;