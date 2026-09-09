---
layout: home
title: Contributing
permalink: /CONTRIBUTING/
---

# Contributing to ucsa

Thanks for your interest in ucsa. This document explains how to set up
the project locally, run the test suite, and submit a pull request.

## Reporting issues

Open an issue at
[`sachncs/ucsa/issues`](https://github.com/sachncs/ucsa/issues)
using the appropriate template (`bug`, `feature_request`, or `question` via
Discussions if enabled). For security issues, follow
[`SECURITY.md`](./SECURITY.md).

## Development setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Tests

```
pytest -q
```

## Lint / format

```
ruff check .
ruff format --check .
```

## Pull request flow

1. Fork the repository.
2. Create a topic branch off `master` (use linear history).
3. Make focused commits with clear messages.
4. Ensure `tests`, `lint`, and `format` all pass.
5. Use the [PR template](./.github/PULL_REQUEST_TEMPLATE.md).
6. Push the branch and open a pull request targeting `master`.

By submitting a pull request, you agree to follow the
[Code of Conduct](./CODE_OF_CONDUCT.md).
