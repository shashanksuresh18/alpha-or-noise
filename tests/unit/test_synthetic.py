"""Unit tests for the acceptance-test data generators in `tests/acceptance/synthetic.py`."""

import numpy as np
import pandas as pd
import pytest

from tests.acceptance import synthetic as syn

pytestmark = pytest.mark.unit

# Tolerance for "mean realised annual Sharpe over the 20 seeds is close to its target":
# 4 standard errors of that mean.
#
# For iid normal returns the standard error of one per-period Sharpe estimate is
# sqrt((1 + SR^2 / 2) / n_obs) (Lo, 2002, "The Statistics of Sharpe Ratios", iid case).
# Annualised and averaged over n_samples independent draws:
#   se = sqrt(252 * (1 + SR^2 / 2) / n_obs / n_samples).
#   - one variant, full 10 years, 20 seeds:    se ~ 0.071  -> tolerance ~ 0.28
#   - one variant, a 5-year half, 20 seeds:    se ~ 0.100  -> tolerance ~ 0.40
#   - 55 cliff variants, 10 years, 20 seeds:   se ~ 0.0095 -> tolerance ~ 0.04
# Why 4 SE: the seeds are fixed, so the test passes or fails deterministically; 4 SE means a
# correct generator would fail on fewer than 1 in 15,000 seed sets, so a failure points at a
# bug. It is still tight enough to catch the mistakes that matter: a missing sqrt(252)
# (Sharpe off ~16x), mean/vol scaling errors, the edge in the wrong column or wrong half.
TOLERANCE_SE = 4.0


def annual_sharpe(r: pd.DataFrame | pd.Series) -> pd.Series | float:
    return r.mean() / r.std(ddof=1) * np.sqrt(syn.PERIODS_PER_YEAR)


def tolerance(target: float, n_obs: int, n_samples: int) -> float:
    sr = target / np.sqrt(syn.PERIODS_PER_YEAR)
    se = np.sqrt(syn.PERIODS_PER_YEAR * (1 + sr**2 / 2) / n_obs / n_samples)
    return TOLERANCE_SE * se


GENERATORS = {
    "A": syn.pure_noise,
    "B": syn.planted_edge,
    "C": syn.train_only_edge,
    "D": lambda seed: syn.parameter_cliff(seed)[0],
}
EXPECTED_N = {"A": 1_000, "B": 1_000, "C": 50, "D": 56}


@pytest.mark.parametrize("name", GENERATORS)
def test_same_seed_identical_output(name: str) -> None:
    gen = GENERATORS[name]
    pd.testing.assert_frame_equal(gen(7), gen(7))
    assert not gen(7).equals(gen(8))


def test_parameter_cliff_params_deterministic() -> None:
    pd.testing.assert_frame_equal(syn.parameter_cliff(3)[1], syn.parameter_cliff(3)[1])


@pytest.mark.parametrize("name", GENERATORS)
def test_shape_index_and_column_order(name: str) -> None:
    r = GENERATORS[name](0)
    n = EXPECTED_N[name]
    assert r.shape == (syn.T, n) == (2_520, n)
    assert list(r.columns) == [f"variant_{i:04d}" for i in range(n)]
    assert isinstance(r.index, pd.DatetimeIndex)
    assert r.index.is_monotonic_increasing and r.index.is_unique
    assert r.notna().all().all()


def test_parameter_cliff_params_layout() -> None:
    returns, params = syn.parameter_cliff(0)
    assert list(params.index) == list(returns.columns)
    assert list(params.columns) == ["lookback"]
    assert params["lookback"].tolist() == list(range(5, 61))


def _seed_mean_sharpe(frames: list[pd.Series]) -> float:
    return float(np.mean([annual_sharpe(f) for f in frames]))


def test_planted_edge_sharpe_b() -> None:
    edge = [syn.planted_edge(s)["variant_0000"] for s in syn.SEEDS]
    target = syn.PLANTED_SHARPE_B
    assert abs(_seed_mean_sharpe(edge) - target) < tolerance(target, syn.T, len(edge))


def test_train_only_edge_sharpe_c() -> None:
    half = syn.TRAIN_YEARS_C * syn.PERIODS_PER_YEAR
    v0 = [syn.train_only_edge(s)["variant_0000"] for s in syn.SEEDS]
    first = [r.iloc[:half] for r in v0]
    second = [r.iloc[half:] for r in v0]
    target = syn.TRAIN_SHARPE_C
    assert abs(_seed_mean_sharpe(first) - target) < tolerance(target, half, len(v0))
    assert abs(_seed_mean_sharpe(second)) < tolerance(0.0, syn.T - half, len(v0))


def test_parameter_cliff_sharpe_d() -> None:
    peak_col = f"variant_{syn.CLIFF_PEAK_LOOKBACK - 5:04d}"
    frames = [syn.parameter_cliff(s)[0] for s in syn.SEEDS]
    peak = [f[peak_col] for f in frames]
    target = syn.CLIFF_PEAK_SHARPE
    assert abs(_seed_mean_sharpe(peak) - target) < tolerance(target, syn.T, len(peak))

    base = np.concatenate([annual_sharpe(f.drop(columns=peak_col)).to_numpy() for f in frames])
    target = syn.CLIFF_BASE_SHARPE
    assert abs(base.mean() - target) < tolerance(target, syn.T, base.size)
