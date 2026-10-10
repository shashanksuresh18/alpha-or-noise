# PROGRESS

Last updated: 2026-10-04 by Claude (planning chat in the Claude app)

## Status
Planning complete. No code yet. Repo contains only `AGENTS.md`, `CLAUDE.md`, `PROGRESS.md` and `docs/detector-v1-spec.md`.

## Next task
**001 — Repo scaffold.** See the queue below.

## Queue

**001 — Repo scaffold**
- `pyproject.toml`: Python 3.12+, numpy, pandas, scipy; dev: pytest, ruff.
- Layout from `AGENTS.md`, with empty `__init__.py` files.
- `.gitignore`: `data/`, `.env`, virtualenvs, notebook checkpoints, caches.
- pytest markers `unit` and `acceptance` registered in `pyproject.toml`.
- GitHub Actions workflow: ruff + unit tests on every push and pull request.
- Done when: CI is green on one trivial test.

**002 — Acceptance tests A–D, written before the detector**
- Synthetic generators in `tests/acceptance/synthetic.py`, exactly as specified in `docs/detector-v1-spec.md` (T = 2,520 daily, daily vol 1%, seeds 0–19).
- Tests import the detector's public API (`alpha_or_noise.detector.evaluate`) and fail because it doesn't exist yet.
- Done when: the four tests exist and fail for that reason, not for any other.

**003 — Sharpe statistics** (`detector/sharpe.py`)
- Sharpe, skewness, kurtosis, PSR, MinTRL, DSR — per-period units throughout.
- Unit tests: one tiny example checked by hand or in a spreadsheet for each formula; PSR of a zero-mean series ≈ 0.5; DSR falls as the number of trials rises.
- Done when: unit tests pass and the other agent has reviewed it.

**004 — PBO via CSCV** (`detector/pbo.py`)
- Block-sum vectorisation as described in the spec.
- Unit tests: a dominant strategy gives PBO = 0; a constructed case where the in-sample winner is always the out-of-sample loser gives PBO = 1; a pure-noise matrix gives PBO near 0.5.
- Done when: unit tests pass and the other agent has reviewed it.

**005 — Robustness checks, verdict and report**
- Walk-forward decay, subperiod stability (ratio rule plus the half-vs-half significance test), parameter cliff, cost sensitivity.
- Verdict rules, `StrategyReport` dataclass, plain-text rendering.
- `evaluate()` wires everything together.
- Done when: acceptance tests A–D pass on at least 19 of 20 seeds each, results are close to the calibration baseline in the spec, and the other agent has reviewed it.

## Later
- 006 — Minimal backtest engine: signal → positions (t+1 execution) → gross and net returns with costs.
- 007 — Append-only trial log (parquet), read by the detector.
- 008 — Effective number of trials (cluster correlated variants) for DSR v2.
- Project 2 data decision: by December 2026.

## In progress
None.

## Unreviewed code
None.

## Decisions log
- 2026-10-04 — Four projects locked, in order: detector → cross-sectional alpha → slow news diffusion → strategy lab.
- 2026-10-04 — Prediction-market execution out of scope (Gambling Commission statement, 4 Feb 2026).
- 2026-10-04 — Claude Code builds, Codex reviews. Roles swap when usage limits hit. Detector and engine maths are never merged unreviewed.
- 2026-10-04 — Forward testing checks operations only. An edge must be established by historical out-of-sample results and the detector.
- 2026-10-04 — Subperiod check includes a half-vs-half significance test; simulation showed the ratio rule alone missed about 1 in 40 decayed edges.

## Open questions
- Python environment tool (uv or pip + venv): the agent doing task 001 decides and records the choice here.
