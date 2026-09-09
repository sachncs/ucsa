---
layout: default
title: Contributing
permalink: /docs/contributing/
nav_order: 5
---

# Contributing

Thanks for considering a contribution. This page covers the
developer workflow, the lint and type-check configuration, the
test suite, the PR template, and the code review checklist.

## Code of conduct

This project follows the Contributor Covenant. By participating,
you agree to abide by its terms. See
[CODE_OF_CONDUCT.md](https://github.com/sachncs/sachncs.github.io/blob/master/CODE_OF_CONDUCT.md).

## Reporting security issues

Please **do not** file a public issue. Use GitHub Security
Advisories instead:
[sachncs/sachncs.github.io/security/advisories/new](https://github.com/sachncs/sachncs.github.io/security/advisories/new).
See [SECURITY.md](https://github.com/sachncs/sachncs.github.io/blob/master/SECURITY.md)
for the supported-versions matrix and the response SLA.

## Filing a non-security issue

Use the appropriate issue template:

- **Bug reports** — `.github/ISSUE_TEMPLATE/bug.yml`. Reproducible
  steps, expected behaviour, environment, logs.
- **Feature requests** — `.github/ISSUE_TEMPLATE/feature_request.yml`.
  Problem statement, proposed solution, alternatives considered,
  additional context.
- **Questions** — please use
  [GitHub Discussions](https://github.com/sachncs/sachncs.github.io/discussions)
  rather than the issue tracker.

Blank issues are disabled; the templates are the canonical entry
point.

## Development setup

```bash
git clone https://github.com/sachncs/sachncs.github.io.git
cd ucsa

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Install the pre-commit hooks (optional but recommended).
pre-commit install
```

## Lint, format, type-check

```bash
ruff check ucsa tests scripts       # lint
ruff check --fix ucsa tests scripts  # autofix safe violations
black --check ucsa tests scripts     # format check
black ucsa tests scripts             # format in place
mypy ucsa                            # type-check
```

`ruff` enforces the rule sets: `E`, `W`, `F`, `I`, `B`, `UP`,
`SIM`, `C4`, `RET`, `YTT`, `PT`. We ignore `E501` (line length is
handled by `black`) and `B008` (function call in default argument,
which Hydra / OmegaConf uses extensively). The per-file overrides
in `pyproject.toml` keep the rules from fighting the test style.

## Tests

```bash
pytest -q                    # everything except the slow marker
pytest -q -m "not slow"      # fast tests only (~600, ~2 minutes)
pytest -q -m slow            # the slow localisation-claim run (~2 minutes)
pytest -q --cov=ucsa         # with coverage
```

Coverage is enforced at 80% in CI; the threshold lives in
`pyproject.toml` under `[tool.coverage.report]`.

The test suite uses a `FakeTokenizer` that maps characters to token
ids, so most tests run offline. The eval harness tests are offline
too; the streaming loaders are monkey-patched to plain iterables.
Tests that actually need network access are marked `integration`
and excluded from the default `pytest -q` invocation.

## Smoke test before submitting a PR

```bash
.venv/bin/python scripts/train.py \
    --max-steps 5 \
    --ckpt-every 0 \
    --eval-every 0 \
    --skip-baselines \
    --seed 42
```

Should exit in under a minute and write `runs/ucsa-<tag>-seed42.json`.

## Pull request flow

1. Fork the repository.
2. Create a branch off `master`:
   `git checkout -b feat/my-change`
3. Make your change. Add a test under `tests/` that exercises the
   new behaviour. Update `docs/` and `TODO.md` if the change
   affects the public surface or the roadmap.
4. Run the lint, format, type-check, and test commands above.
5. Push your branch and open a PR against `master`. Fill in the
   PR template (`.github/PULL_REQUEST_TEMPLATE.md`).
6. Wait for CI. The `test` job runs on Python 3.11 and 3.12; the
   `slow` job runs the localisation-claim test on Python 3.12.
7. Address review feedback. The reviewer may ask for additional
   tests, a docs update, or a clarification in the commit
   message.

## Commit message style

This project uses Conventional Commits:

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

Types in use:

- `feat` — new feature
- `fix` — bug fix
- `test` — adding or fixing tests
- `docs` — documentation only
- `refactor` — code change that neither fixes a bug nor adds a
  feature
- `chore` — toolchain, CI, dependencies, housekeeping

Examples:

```
fix(trainer): pad short targets with ignore_index, not 0
docs(readme): link TODO.md from the project roadmap
refactor(state): drop ponytail markers from production code
```

## Code review checklist

Reviewers look for:

- **Correctness** — does the change do what the PR claims? Are
  edge cases covered?
- **Tests** — is the new behaviour exercised? Are regressions
  caught?
- **Docs** — are user-facing changes reflected in `docs/`,
  `README.md`, `CHANGELOG.md`, and `TODO.md`?
- **Style** — does the change follow the conventions in this
  file and in `pyproject.toml`?
- **Performance** — does the change introduce a hot-path
  allocation or a per-step dictionary lookup that should be a
  module-level constant?

## Project layout

```
ucsa/
├── configs/         Hydra / OmegaConf YAML configuration
├── models/          PCS, operators, memory, heads, losses, top-level UCSA
├── training/        trainer, dataset, curriculum, metrics,
│                    evaluation, EMA, Muon optimiser, eval harness
├── utils/           seed, checkpoint, logging
├── tests/           pytest suite
├── train.py         training entrypoint
├── infer.py         inference entrypoint
└── paper/           paper draft (PAPER.md, TABLES.md)
scripts/
├── train.py         UCSA training + SOTA stack + benchmark comparison
├── train_baseline.py matched-compute vanilla-Transformer baseline
├── eval.py          HellaSwag / ARC / PIQA / WinoGrande evaluation
├── probe_banks.py   PCS bank probe (top tokens per bank, centroid sim)
├── probe_origination.py intent-bank localisation, collapse, and descent probes
├── run_ablations.py ablation matrix driver
├── build_paper_tables.py reads runs/*.json, writes paper/TABLES.md
└── benchmark.py     one-file showcase against modern-LM baselines
docs/
├── index.md         Jekyll landing page
├── getting-started.md install, configure, smoke-test
├── architecture.md  deep design notes
├── api-reference.md module-by-module API tour
├── tutorials.md     end-to-end walkthroughs
└── contributing.md  this page
```

## Where to ask questions

- **GitHub Discussions** — open-ended questions, design
  discussions, "how do I…?" posts.
- **GitHub Issues** — bugs, feature requests, with templates.
- **Email** — `sachncs@gmail.com` for security or private
  correspondence.

## Where to go next

- [Architecture →](architecture.md) for the design notes behind
  the API.
- [Tutorials →](tutorials.md) for end-to-end walkthroughs.
