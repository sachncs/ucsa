# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding: `pyproject.toml` with project metadata, dependencies,
  and tool configuration for ruff, black, mypy, and pytest.
- README.md with quickstart, layout, and extension guide.
- docs/architecture.md with deep design notes (PCS contract, memory pipeline,
  JEPA cadence, verifier signal flow).
- TODO.md tracking every atomic commit through Phase 10.
- Package skeleton: `ucsa/`, `ucsa/models/`, `ucsa/training/`, `ucsa/utils/`,
  `ucsa/configs/`, `tests/`.
- `ucsa.models.state`: `PersistentCognitiveState`, `PCSConfig`, `BankSpec`,
  `retention_score`, `resolve_bank_sizes`, `build_bank_specs`. Six token
  banks (working/long_term/goal/episode/task/memory_index), retention
  metadata buffers, recycle policy.