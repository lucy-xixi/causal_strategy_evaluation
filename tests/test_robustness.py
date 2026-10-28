import numpy as np
import pandas as pd
import pytest

from src.config import POLICY_START_MONTH, REGION_TYPES
from src.data_generation import generate_panel
from src.data_quality import copy_with_issue
from src.robustness import (
    ASSIGNMENT_RULE,
    draw_placebo_labels,
    placebo_time_check,
    run_robustness,
    window_check,
)


@pytest.fixture(scope="module")
def noisy_panel() -> pd.DataFrame:
    return generate_panel(8)


@pytest.fixture(scope="module")
def structural_panel() -> pd.DataFrame:
    return generate_panel(8, structural=True)


def test_robustness_suite_records_windows_and_does_not_modify_data(noisy_panel):
    snapshot = noisy_panel.copy(deep=True)
    suite = run_robustness(noisy_panel, placebo_reps=4, placebo_seed=11)
    pd.testing.assert_frame_equal(noisy_panel, snapshot)
    by_name = {check.name: check for check in suite.checks}
    baseline = by_name["baseline_twfe"]
    assert baseline.status == "success"
    assert baseline.n_obs == len(noisy_panel)
    assert baseline.n_regions == 48
    assert baseline.n_treated_regions == 20

    wide = by_name["window_pre6_post6"]
    narrow = by_name["window_pre3_post3"]
    assert wide.status == "success"
    assert narrow.status == "success"
    assert (wide.month_min, wide.month_max) == (POLICY_START_MONTH - 6, POLICY_START_MONTH + 5)
    assert (narrow.month_min, narrow.month_max) == (POLICY_START_MONTH - 3, POLICY_START_MONTH + 2)
    assert wide.n_obs == noisy_panel["region_id"].nunique() * 12
    assert narrow.n_obs == noisy_panel["region_id"].nunique() * 6
    assert wide.n_regions == baseline.n_regions
    assert "是否显著不能单独作为稳健性的结论" in wide.comparison_to_baseline
    frame = suite.to_frame()
    assert set(frame.loc[frame["status"] == "success", "name"]) >= {
        "baseline_twfe",
        "window_pre6_post6",
        "window_pre3_post3",
        "covariate_promo_intensity",
    }


def test_invalid_window_fails_without_dropping_rows(noisy_panel):
    snapshot = noisy_panel.copy(deep=True)
    failed = window_check(noisy_panel, None, pre_months=0, post_months=6)
    assert failed.status == "failed"
    assert failed.estimate is None
    assert "处理前" in failed.failure_reason
    outside = window_check(noisy_panel, None, pre_months=100, post_months=6)
    assert outside.status == "failed"
    assert outside.estimate is None
    pd.testing.assert_frame_equal(noisy_panel, snapshot)


def test_placebo_time_excludes_post_policy_rows(noisy_panel, structural_panel):
    check = placebo_time_check(noisy_panel)
    assert check.status == "success"
    assert check.policy_start_month == POLICY_START_MONTH
    assert check.placebo_policy_start is not None
    assert check.placebo_policy_start < POLICY_START_MONTH
    assert check.month_max < POLICY_START_MONTH
    assert check.month_min >= 0
    pre_months = noisy_panel.loc[noisy_panel["month"] < POLICY_START_MONTH, "month"].nunique()
    assert check.n_obs == noisy_panel["region_id"].nunique() * pre_months
    assert check.n_treated_regions == 20
    structural = placebo_time_check(structural_panel)
    assert structural.status == "success"
    assert structural.estimate == pytest.approx(0.0, abs=1e-6)


def test_covariate_check_uses_promo_and_explains_region_baseline(noisy_panel, structural_panel):
    suite = run_robustness(noisy_panel, placebo_reps=2, placebo_seed=3)
    check = next(item for item in suite.checks if item.name == "covariate_promo_intensity")
    assert check.status == "success"
    assert check.n_obs == len(noisy_panel)
    assert check.n_regions == 48
    assert check.estimate is not None and np.isfinite(check.estimate)
    assert "region_baseline" in check.interpretation
    assert "吸收" in check.interpretation
    absorbed = next(
        item
        for item in run_robustness(structural_panel, placebo_reps=2, placebo_seed=3).checks
        if item.name == "covariate_promo_intensity"
    )
    assert absorbed.status == "failed"
    assert absorbed.estimate is None
    assert "剩余" in absorbed.failure_reason


def test_placebo_assignment_preserves_counts_and_seed(noisy_panel):
    snapshot = noisy_panel.copy(deep=True)
    suite = run_robustness(noisy_panel, placebo_reps=5, placebo_seed=19)
    again = run_robustness(noisy_panel, placebo_reps=5, placebo_seed=19)
    pd.testing.assert_frame_equal(noisy_panel, snapshot)
    check = next(item for item in suite.checks if item.name == "placebo_assignment")
    repeat = next(item for item in again.checks if item.name == "placebo_assignment")
    assert check.status == "success"
    assert check.n_replications == 5
    assert check.placebo_seed == 19
    assert len(check.placebo_estimates) == 5
    assert check.placebo_estimates == repeat.placebo_estimates
    assert check.p_value is None
    assert check.p_value_reason
    assert check.n_treated_regions == 20
    assert "每个 region_type" in check.assignment_rule
    assert check.assignment_rule == ASSIGNMENT_RULE

    rng = np.random.default_rng(19)
    labels = draw_placebo_labels(noisy_panel, rng)
    counts = (
        noisy_panel.drop_duplicates("region_id")
        .assign(placebo=lambda frame: frame["region_id"].map(labels))
        .groupby("region_type")[["treated", "placebo"]]
        .sum()
    )
    assert counts["treated"].to_dict() == counts["placebo"].to_dict()
    for spec in REGION_TYPES:
        assert counts.loc[spec.name, "placebo"] == spec.n_treated
    manual = noisy_panel.copy(deep=True)
    manual["treated"] = manual["region_id"].map(labels).astype(int)
    manual["treated_post"] = manual["treated"] * manual["post"].astype(int)
    manual["treatment_start_month"] = pd.Series(pd.NA, index=manual.index, dtype="Int64")
    manual.loc[manual["treated"] == 1, "treatment_start_month"] = POLICY_START_MONTH
    from src.did_analysis import fit_did

    assert check.placebo_estimates[0] == pytest.approx(fit_did(manual).estimate)


def test_illegal_input_raises_and_leaves_original_unchanged(noisy_panel):
    snapshot = noisy_panel.copy(deep=True)
    missing = copy_with_issue(noisy_panel, "missing_outcome")
    with pytest.raises(ValueError, match="缺失值"):
        run_robustness(missing, placebo_reps=1, placebo_seed=1)
    illegal = copy_with_issue(noisy_panel, "illegal_interaction")
    with pytest.raises(ValueError, match="treated_post"):
        run_robustness(illegal, placebo_reps=1, placebo_seed=1)
    pd.testing.assert_frame_equal(noisy_panel, snapshot)
    assert len(missing) == len(noisy_panel) + 0 or len(missing) == len(noisy_panel)
