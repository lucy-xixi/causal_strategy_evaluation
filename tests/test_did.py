import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from src.config import tau_by_type
from src.data_generation import generate_panel
from src.data_quality import copy_with_issue
from src.did_analysis import fit_did
from tests.test_data_generation import (
    ALL_REGION_WEIGHTED,
    HAND_TARGET,
    UNWEIGHTED_TYPE_MEAN,
)

# 单次估计的标准误大约是 0.27。20 次重复的均值标准误大约是 0.06，
# 所以 0.25 大约是均值抽样波动的 4 倍，用来检查方法是否围绕 2.8，而不是卡住某一次抽样。
MONTE_CARLO_SEEDS = range(20)
MEAN_TOLERANCE = 0.25


def test_structural_did_matches_treated_region_average_across_seeds():
    for seed in (0, 1, 2, 7, 42):
        panel = generate_panel(seed, structural=True)
        result = fit_did(panel, tau_by_type=tau_by_type())
        assert result.estimand_matches_target
        assert result.target_effect == pytest.approx(HAND_TARGET)
        assert result.estimate == pytest.approx(HAND_TARGET, abs=1e-6)
        assert result.bias == pytest.approx(0.0, abs=1e-6)
        assert result.target_effect != pytest.approx(UNWEIGHTED_TYPE_MEAN)


def test_month_effects_are_required_for_the_target():
    panel = generate_panel(4, structural=True)
    result = fit_did(panel, tau_by_type=tau_by_type())
    wrong = smf.ols(
        "outcome ~ treated_post + C(region_id)",
        data=panel,
        missing="raise",
    ).fit()
    wrong_error = abs(float(wrong.params["treated_post"]) - HAND_TARGET)
    assert wrong_error > abs(result.estimate - HAND_TARGET) + 1.0


def test_result_reports_coefficient_uncertainty_and_sample():
    panel = generate_panel(42)
    snapshot = panel.copy(deep=True)
    result = fit_did(panel, tau_by_type=tau_by_type())
    pd.testing.assert_frame_equal(panel, snapshot)

    assert result.covariance_type == "cluster"
    assert result.specification == "outcome ~ treated_post + C(region_id) + C(month)"
    substantive = [
        name for name in result.exog_names if name != "Intercept" and not name.startswith("C(")
    ]
    assert substantive == ["treated_post"]
    assert result.std_error > 0
    assert result.ci_low < result.estimate < result.ci_high
    assert 0 <= result.p_value <= 1
    expected_half = result.t_critical * result.std_error
    assert result.ci_high - result.estimate == pytest.approx(expected_half)
    assert result.ci_low == pytest.approx(result.estimate - expected_half)
    assert result.t_critical == pytest.approx(float(stats.t.ppf(0.975, result.n_regions - 1)))
    assert result.t_df == 47
    assert result.n_obs == len(panel) == 48 * 24
    assert result.n_dropped == 0
    assert result.n_regions == 48
    assert result.n_treated_regions == 20
    assert result.n_control_regions == 28
    assert result.n_pre_months == 12
    assert result.n_post_months == 12
    assert (result.month_min, result.month_max) == (0, 23)
    assert result.policy_start_month == 12
    assert result.target_effect == pytest.approx(HAND_TARGET)
    assert result.bias == pytest.approx(result.estimate - HAND_TARGET)
    assert any("聚类" in warning and "不能视为已经解决" in warning for warning in result.warnings)


def test_row_order_does_not_change_the_estimate():
    panel = generate_panel(8)
    shuffled = panel.sample(frac=1.0, random_state=0)
    original = fit_did(panel, tau_by_type=tau_by_type())
    reordered = fit_did(shuffled, tau_by_type=tau_by_type())
    assert reordered.estimate == pytest.approx(original.estimate, abs=1e-8)
    assert reordered.std_error == pytest.approx(original.std_error, rel=1e-6)


def test_estimator_does_not_repair_illegal_samples():
    panel = generate_panel(6)
    snapshot = panel.copy(deep=True)
    missing = copy_with_issue(panel, "missing_outcome")
    duplicate = copy_with_issue(panel, "duplicate_region_month")
    illegal = copy_with_issue(panel, "illegal_interaction")
    pre_only = panel.loc[panel["post"] == 0].copy()
    with pytest.raises(ValueError, match="缺失值"):
        fit_did(missing)
    with pytest.raises(ValueError, match="重复"):
        fit_did(duplicate)
    with pytest.raises(ValueError, match="treated_post"):
        fit_did(illegal)
    with pytest.raises(ValueError, match="处理前和处理后"):
        fit_did(pre_only)
    pd.testing.assert_frame_equal(panel, snapshot)
    assert len(missing) == len(panel)


def test_unbalanced_copy_is_estimated_without_claiming_the_target_bias():
    panel = generate_panel(6, structural=True)
    unbalanced = panel.drop(index=panel.index[0]).copy()
    result = fit_did(unbalanced, tau_by_type=tau_by_type())
    assert result.n_obs == len(unbalanced)
    assert result.n_dropped == 0
    assert result.target_effect == pytest.approx(HAND_TARGET)
    assert result.estimand_matches_target is False
    assert result.bias is None


def test_repeated_simulations_center_on_the_treated_region_target():
    estimates = []
    standard_errors = []
    covered = []
    slopes = []
    for seed in MONTE_CARLO_SEEDS:
        panel = generate_panel(seed)
        result = fit_did(panel, tau_by_type=tau_by_type())
        estimates.append(result.estimate)
        standard_errors.append(result.std_error)
        covered.append(result.ci_low <= HAND_TARGET <= result.ci_high)
        slopes.append(_pretrend_slope(panel))

    mean_estimate = float(np.mean(estimates))
    assert abs(mean_estimate - HAND_TARGET) < MEAN_TOLERANCE
    assert abs(mean_estimate - HAND_TARGET) < abs(mean_estimate - UNWEIGHTED_TYPE_MEAN)
    assert abs(mean_estimate - HAND_TARGET) < abs(mean_estimate - ALL_REGION_WEIGHTED)
    assert 0.05 < float(np.median(standard_errors)) < 1.5
    assert np.mean(covered) >= 0.70
    assert abs(float(np.mean(slopes))) < 0.10


def _pretrend_slope(df: pd.DataFrame) -> float:
    pre = df[df["post"] == 0]
    means = pre.groupby(["month", "treated"], as_index=False)["outcome"].mean()
    wide = means.pivot(index="month", columns="treated", values="outcome").sort_index()
    gap = (wide[1] - wide[0]).to_numpy(dtype=float)
    x = wide.index.to_numpy(dtype=float)
    x = x - x.mean()
    return float(np.sum(x * (gap - gap.mean())) / np.sum(x**2))
