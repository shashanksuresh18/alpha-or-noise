# Detector v1 — specification

## Purpose

Given **every** strategy variant that was tried, decide whether the best one shows evidence that survives selection bias, or is probably noise.
Default stance is hostile: `REJECT` unless the evidence clears every bar.

## Inputs

| Name | Type | Required | Notes |
|---|---|---|---|
| `returns` | DataFrame, T × N | yes | DatetimeIndex. One column per variant tried, failures included. Net-of-cost per-period returns. |
| `periods_per_year` | int | yes | 252 for daily, 12 for monthly. |
| `params` | DataFrame, N rows | no | Variant id → parameter values. Needed for the parameter-cliff check. |
| `turnover` | DataFrame, T × N | no | Needed for cost sensitivity. |
| `config` | dataclass | no | Thresholds; defaults below. |

The selected strategy is the column with the highest full-sample Sharpe, unless the caller names one.
Checks whose optional input is missing are reported as **not run**. A `CANDIDATE` verdict with any check not run does not clear the live-money gate.

## Calculations

All Sharpe-based statistics use **per-period (non-annualised)** Sharpe ratios. Annualise only for display.
SR = selected strategy's per-period Sharpe; γ3 = skewness; γ4 = kurtosis (non-excess; 3 for a normal distribution); T = number of observations.

1. **Sharpe, γ3, γ4** of the selected strategy.

2. **Probabilistic Sharpe Ratio** (Bailey & López de Prado, 2012):

   PSR(SR\*) = Φ( (SR − SR\*) · √(T − 1) / √(1 − γ3·SR + (γ4 − 1)/4 · SR²) )

3. **Minimum Track Record Length** at confidence 1 − α (default 95%):

   MinTRL = 1 + (1 − γ3·SR + (γ4 − 1)/4 · SR²) · ( z₁₋α / (SR − SR\*) )²

   Report in periods and in years. Rule of thumb for normal returns: about (1.645 / annual Sharpe)² years to confirm SR > 0.

4. **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014): DSR = PSR(SR₀), where

   SR₀ = √V · ( (1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e)) )

   V = variance of the per-period Sharpe ratios across all N variants; γ ≈ 0.5772 (Euler–Mascheroni); e = Euler's number.

5. **Probability of Backtest Overfitting via CSCV** (Bailey, Borwein, López de Prado & Zhu):
   - Drop leading rows so T divides evenly, then split rows into S contiguous blocks (S even; default 16).
   - For each of the C(S, S/2) block combinations: in-sample (IS) = chosen blocks, out-of-sample (OOS) = the rest.
   - n\* = variant with the highest IS Sharpe. ω = OOS rank of n\* / (N + 1), ranks 1…N ascending. λ = ln(ω / (1 − ω)).
   - **PBO** = share of combinations with λ ≤ 0.
   - Also report the λ distribution and the slope of OOS Sharpe regressed on IS Sharpe for n\*.
   - Performance: precompute per-block sums and sums of squares for every column (S × N), so each combination's Sharpe ratios are additions, not passes over raw data.

6. **Parameter cliff** (needs `params`): compare the selected variant's Sharpe with the variants one step away in each parameter. **Flag** if the neighbours' median Sharpe is below 50% of the selected Sharpe.

7. **Walk-forward decay**: split rows into K = 5 sequential folds. For k = 2…5, pick the best variant on folds 1…k−1 (expanding window) and record its Sharpe on fold k. **Flag** if the mean OOS Sharpe over the last two folds is below 50% of the selected strategy's full-sample Sharpe.

8. **Subperiod stability**: Sharpe of the selected strategy in each half of the sample and in each calendar year. **Flag** if either condition holds:
   - either half's Sharpe is below 50% of the full-sample Sharpe; or
   - the second half is significantly worse than the first: z = (SR₁ − SR₂) / √(se₁² + se₂²) > 2.33, where se_i = √( (1 − γ3·SR_i + (γ4 − 1)/4 · SR_i²) / (T_i − 1) ) uses each half's own moments.

   The ratio rule alone let roughly 1 in 40 train-only edges through in simulation; the significance test closes that gap.

9. **Cost sensitivity** (needs `turnover`): recompute net Sharpe at 0.5×, 1×, 2× and 3× the default cost. **Flag** if Sharpe ≤ 0 at 2×.

## Verdict rules (provisional defaults, configurable)

- `REJECT` if DSR < 0.90, or PBO > 0.50.
- `CANDIDATE` if DSR ≥ 0.95, PBO ≤ 0.25, and no flags among the checks that ran.
- `INVESTIGATE` otherwise.

The report lists every reason behind the verdict and every check not run.

## Output

A `StrategyReport` dataclass holding every number above plus the verdict and reasons, with a plain-text rendering:

```text
STRATEGY REPORT — variant_0412        verdict: REJECT
Sharpe (annual)        1.04      MinTRL      2.5 years
PSR (vs 0)             0.999     DSR         0.48
PBO                    0.52      trials      1,000
Walk-forward           FLAG      Subperiods  no flag
Parameter cliff        not run   Costs       not run
Reasons: DSR below 0.90; PBO above 0.50; walk-forward decay.
```

## Acceptance tests

Synthetic data only, in `tests/acceptance/`, marked `acceptance`.
Default shape: 10 years of daily data (T = 2,520, 252 periods per year), iid normal returns.
**Each scenario runs across 20 seeds and must give the expected result in at least 19.** A single lucky seed proves nothing — the same logic the detector applies to strategies.

| Test | Construction | Expected |
|---|---|---|
| **A. Pure noise** | 1,000 zero-mean variants. | Verdict `REJECT`; DSR < 0.95. |
| **B. Planted edge** | 999 zero-mean variants + 1 with annual Sharpe 2.5. | Planted variant selected; verdict `CANDIDATE`; PBO ≤ 0.25. |
| **C. Train-only edge** | 50 variants; variant 0 has annual Sharpe 3.0 in years 1–5 and zero mean in years 6–10; the rest zero mean. | Verdict is **not** `CANDIDATE`; walk-forward or subperiod check flags. |
| **D. Parameter cliff** | 56 variants indexed by lookback 5…60; lookback 20 has annual Sharpe 2.6, all others 0.35. `params` supplied. | Parameter-cliff flag raised; verdict `INVESTIGATE`. |

Test C matters because the deflated Sharpe ratio alone can pass a strategy whose edge has disappeared; the time-based checks must catch it.
Test D's edge is real by construction; the check exists because in real data an isolated peak in parameter space is usually overfitting.

**Calibration baseline (4 Oct 2026).** A throwaway reference implementation of this spec, with daily volatility 1% and the shapes above, gave:

| Test | Result | Typical values |
|---|---|---|
| A | 20/20 seeds | best noise variant: annual Sharpe ≈ 1.0, DSR ≈ 0.47, PBO ≈ 0.53 |
| B | 20/20 seeds | DSR ≥ 0.995, PBO ≤ 0.08; time checks false-flag ≈ 1% of seeds |
| C | 20/20 seeds (≈ 99.7% over 300) | PBO median ≈ 0.11 — **CSCV alone does not catch a decayed edge**; the time checks do |
| D | 20/20 seeds | DSR ≈ 1.0, PBO ≈ 0; caught only by the parameter-cliff check |

An implementation that lands far from these numbers probably has a bug.

**If a scenario fails:** suspect the implementation first, the threshold second. Never tune a threshold on a single seed. Record any threshold change, and why, in `docs/journal/`.

Runtime target: the full acceptance suite in under 2 minutes on a laptop.

## Later versions

- Effective number of trials: cluster correlated variants so that 50 near-identical parameter settings are not counted as 50 independent trials.
- White's Reality Check / Hansen's SPA test as a second opinion on selection.
- Reading trials directly from the trial log.
