"""Synthetic return generators for detector acceptance tests A-D.

Constructions follow `docs/detector-v1-spec.md`, "Acceptance tests": 10 years of daily data
(T = 2,520, 252 periods per year), iid normal returns, daily volatility 1%.

A planted annual Sharpe S becomes a per-period mean of ``S / sqrt(252) * DAILY_VOL``, so the
*expected* Sharpe is the target; any single seed's realised Sharpe is noisy around it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

T = 2_520
PERIODS_PER_YEAR = 252
DAILY_VOL = 0.01
SEEDS = range(20)
START = "2010-01-01"

PLANTED_SHARPE_B = 2.5
TRAIN_SHARPE_C = 3.0
TRAIN_YEARS_C = 5
CLIFF_LOOKBACKS = range(5, 61)
CLIFF_PEAK_LOOKBACK = 20
CLIFF_PEAK_SHARPE = 2.6
CLIFF_BASE_SHARPE = 0.35


def variant_ids(n: int) -> list[str]:
    return [f"variant_{i:04d}" for i in range(n)]


def daily_mean(annual_sharpe: float) -> float:
    """Per-period mean giving the target annual Sharpe at DAILY_VOL."""
    return annual_sharpe / np.sqrt(PERIODS_PER_YEAR) * DAILY_VOL


def _frame(mu: np.ndarray, seed: int) -> pd.DataFrame:
    """iid normal returns with per-cell mean ``mu`` (shape T x N)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, DAILY_VOL, size=mu.shape)
    index = pd.bdate_range(START, periods=mu.shape[0], name="date")
    return pd.DataFrame(mu + noise, index=index, columns=variant_ids(mu.shape[1]))


def pure_noise(seed: int, n_variants: int = 1_000) -> pd.DataFrame:
    """Test A: ``n_variants`` zero-mean variants."""
    return _frame(np.zeros((T, n_variants)), seed)


def planted_edge(seed: int, n_variants: int = 1_000) -> pd.DataFrame:
    """Test B: variant 0 has annual Sharpe 2.5; the other 999 are zero mean."""
    mu = np.zeros((T, n_variants))
    mu[:, 0] = daily_mean(PLANTED_SHARPE_B)
    return _frame(mu, seed)


def train_only_edge(seed: int, n_variants: int = 50) -> pd.DataFrame:
    """Test C: variant 0 has annual Sharpe 3.0 in years 1-5, zero mean in years 6-10."""
    mu = np.zeros((T, n_variants))
    mu[: TRAIN_YEARS_C * PERIODS_PER_YEAR, 0] = daily_mean(TRAIN_SHARPE_C)
    return _frame(mu, seed)


def parameter_cliff(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test D: 56 variants, lookback 5..60; lookback 20 has annual Sharpe 2.6, others 0.35.

    Returns ``(returns, params)``. ``params`` is indexed by variant id, with one column
    ``lookback``, in the same order as the ``returns`` columns.
    """
    lookbacks = np.array(CLIFF_LOOKBACKS)
    sharpe = np.where(lookbacks == CLIFF_PEAK_LOOKBACK, CLIFF_PEAK_SHARPE, CLIFF_BASE_SHARPE)
    mu = np.broadcast_to(daily_mean(1.0) * sharpe, (T, lookbacks.size)).copy()
    returns = _frame(mu, seed)
    params = pd.DataFrame({"lookback": lookbacks}, index=pd.Index(returns.columns, name="variant"))
    return returns, params
