"""Detector acceptance tests A-D (`docs/detector-v1-spec.md`, "Acceptance tests").

Each scenario runs on seeds 0-19 and must give the expected result on at least 19.

The detector API these tests pin down:
    evaluate(returns, periods_per_year, params=None, turnover=None, config=None) -> StrategyReport
with report fields ``verdict`` ("REJECT" | "INVESTIGATE" | "CANDIDATE", str or StrEnum),
``selected`` (column name), ``dsr``, ``pbo``, and booleans ``walk_forward_flag``,
``subperiod_flag``, ``parameter_cliff_flag``.

`evaluate` is imported inside each test, not at module level: a module-level ImportError would
be a collection error, which breaks `pytest -m "not acceptance"` even though these tests are
deselected.
"""

from collections.abc import Callable

import pytest

from tests.acceptance import synthetic as syn

pytestmark = pytest.mark.acceptance

MIN_PASSING_SEEDS = 19


def _evaluate() -> Callable:
    from alpha_or_noise.detector import evaluate

    return evaluate


def _assert_enough_seeds(passed: list[int]) -> None:
    failed = sorted(set(syn.SEEDS) - set(passed))
    assert len(passed) >= MIN_PASSING_SEEDS, f"passed {len(passed)}/20; failed seeds {failed}"


def test_a_pure_noise_rejected() -> None:
    evaluate = _evaluate()
    passed = []
    for seed in syn.SEEDS:
        report = evaluate(syn.pure_noise(seed), periods_per_year=syn.PERIODS_PER_YEAR)
        if report.verdict == "REJECT" and report.dsr < 0.95:
            passed.append(seed)
    _assert_enough_seeds(passed)


def test_b_planted_edge_is_candidate() -> None:
    evaluate = _evaluate()
    passed = []
    for seed in syn.SEEDS:
        report = evaluate(syn.planted_edge(seed), periods_per_year=syn.PERIODS_PER_YEAR)
        if (
            report.selected == "variant_0000"
            and report.verdict == "CANDIDATE"
            and report.pbo <= 0.25
        ):
            passed.append(seed)
    _assert_enough_seeds(passed)


def test_c_train_only_edge_not_candidate() -> None:
    evaluate = _evaluate()
    passed = []
    for seed in syn.SEEDS:
        report = evaluate(syn.train_only_edge(seed), periods_per_year=syn.PERIODS_PER_YEAR)
        time_flag = report.walk_forward_flag or report.subperiod_flag
        if report.verdict != "CANDIDATE" and time_flag:
            passed.append(seed)
    _assert_enough_seeds(passed)


def test_d_parameter_cliff_flagged() -> None:
    evaluate = _evaluate()
    passed = []
    for seed in syn.SEEDS:
        returns, params = syn.parameter_cliff(seed)
        report = evaluate(returns, periods_per_year=syn.PERIODS_PER_YEAR, params=params)
        if report.parameter_cliff_flag and report.verdict == "INVESTIGATE":
            passed.append(seed)
    _assert_enough_seeds(passed)
