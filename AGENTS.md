# alpha-or-noise — agent brief

Read this file, then `PROGRESS.md`, at the start of every session.
This is the single source of truth for every coding agent (Claude Code and Codex). `CLAUDE.md` imports it — edit this file, not that one.

## Mission

Build a quant research platform whose job is to find evidence that survives our best attempts to prove it fake.

- **2027 success** = a rigorous research platform, plus at least one strategy with credible evidence of positive expected value after realistic costs.
- A profitable backtest is **not** success. Assume every strategy is noise until the evidence says otherwise.

## Projects, in build order

1. **Detector + backtest engine** (`detector/`, `engine/`) — the referee. Every other project depends on it. Full spec: `docs/detector-v1-spec.md`.
2. **Cross-sectional alpha** (`alpha/`) — monthly-rebalanced stock ranking. Linear baseline first, then gradient boosting. Measure the long-short spread to judge the signal; implement long-only (top bucket vs index), because shorting single stocks is too costly at retail.
3. **Slow news diffusion** (`diffusion/`) — information spreading over days to weeks, via customer–supplier links and changes in filing language. Source: SEC EDGAR, using filing acceptance timestamps. No intraday latency races.
4. **AI strategy search lab** (`lab/`) — proposes variations within research areas that have already been validated. Every result goes through the detector. Built last.

**Out of scope:** prediction-market execution (GB regulation), intraday or latency-sensitive strategies, deep learning before simple models are hard to beat.

## Session protocol

1. Read `AGENTS.md` and `PROGRESS.md`. Work on the task under **Next task** unless the human says otherwise.
2. One task per branch: `task/NNN-short-name`. Never work on a branch listed as in progress by another agent.
3. Tests first: write or confirm the failing test before writing the implementation.
4. Keep each task to about an hour. Split anything bigger and add the pieces to the queue in `PROGRESS.md`.
5. **Before stopping for any reason — finished, interrupted, or near a usage limit:** run the tests, commit, push, and update `PROGRESS.md` with exactly where you stopped and what comes next. If limits are close, do this first: a half-done task with a clear handoff beats a finished task with none.
6. End every commit message with a trailer naming the agent: `Agent: claude-code` or `Agent: codex`.
7. List every unmerged or unreviewed change under **Unreviewed code** in `PROGRESS.md`.

## Review rule

Any change touching the maths in `engine/` or `detector/` must be reviewed by the *other* agent before it is merged to `main`. If the reviewer is unavailable, the merge waits; work continues on the next task in a new branch.

Reviewer checklist:
- **Look-ahead:** does any calculation use data from after its timestamp?
- **Alignment:** a signal computed at the close of bar t trades at bar t+1, never the same bar.
- **Units:** per-period vs annualised Sharpe. PSR, DSR and MinTRL use per-period values.
- **Statistics:** sample vs population standard deviation; `ddof` used consistently.
- **Edge cases:** NaNs, constant columns, empty or uneven partitions.
- **Determinism:** fixed random seeds in all tests.

## Research rules

- **Trial log:** every backtest run is appended to the trial log, failures included. The detector's trial count comes from it. Never delete trials.
- **Locked holdout:** the most recent 2 years of any dataset are reserved. A finalist strategy touches it once, and that touch is logged in `docs/journal/`. The lab never sees it.
- **Costs always on:** every backtest reports gross and net returns. Default cost is 10 bps per side, configurable.
- **Point-in-time only:** every feature must be computable from information available at its timestamp.
- **Free data:** fine for building pipelines. Performance numbers from free data are not evidence, because free sources usually omit delisted stocks.
- **LLMs are extractors, not judges:** LLMs turn text into structured fields; statistical models decide whether those fields predict returns. Predictive tests use only events after the LLM's training cutoff.

## Live-money gate

All of: robust out-of-sample results; detector verdict `CANDIDATE` with every check run; still profitable at 2× costs and at neighbouring parameters; no degradation in forward testing; operations verified end to end. Then start very small.
Forward testing checks operations. It cannot prove an edge — that takes years of live data (see MinTRL in the spec).

## Code conventions

- Python 3.12+, `src/` layout, pytest, ruff, type hints.
- `detector/` depends only on numpy, pandas and scipy.
- Every statistical function's docstring cites its paper and equation.
- Every formula has a small hand-checked unit test.
- `notebooks/` is for exploration only; nothing imports from it.
- `data/` is gitignored. Never commit data, credentials or API keys.
- Slow tests carry the `acceptance` pytest marker so the unit suite stays fast.

## Layout

```text
src/alpha_or_noise/
  engine/      backtesting, costs, data loading
  detector/    Sharpe statistics, PBO/CSCV, robustness checks, verdicts
  alpha/       project 2
  diffusion/   project 3
  lab/         project 4
tests/
  unit/
  acceptance/  detector acceptance tests A–D
docs/
  detector-v1-spec.md
  journal/     research journal and holdout log
notebooks/
data/          gitignored
```
