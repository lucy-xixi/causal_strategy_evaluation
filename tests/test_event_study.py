import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer
from scipy import stats

from src.config import POLICY_START_MONTH, tau_by_type
from src.data_generation import generate_panel
from src.data_quality import copy_with_issue
from src.estimand import design_target
from src.config import REGION_TYPES
from src.event_study import fit_event_study
from src.plotting import plot_event_study

TARGET = design_target(REGION_TYPES)


@pytest.fixture(scope="module")
def structural_panel() -> pd.DataFrame:
    return generate_panel(4, structural=True)


@pytest.fixture(scope="module")
def noisy_panel() -> pd.DataFrame:
    return generate_panel(4)


def test_event_study_is_reproducible(noisy_panel):
    first = fit_event_study(noisy_panel)
    second = fit_event_study(noisy_panel)
    assert first.reference_event_time == -1
    assert [(item.event_time, item.estimate) for item in first.estimates] == [
        (item.event_time, item.estimate) for item in second.estimates
    ]


def test_reference_period_is_omitted_and_not_reported_as_zero(noisy_panel):
    result = fit_event_study(noisy_panel)
    reference = next(item for item in result.estimates if item.event_time == result.reference_event_time)
    assert reference.estimated is False
    assert reference.role == "reference"
    assert reference.estimate is None
    assert reference.std_error is None
    assert reference.ci_low is None
    assert reference.ci_high is None
    assert reference.p_value is None
    assert "不是估计值为零" in reference.note
    assert "统一处理时点" in result.scope_note
    assert result.policy_start_month == POLICY_START_MONTH


def test_estimated_periods_have_complete_intervals(noisy_panel):
    result = fit_event_study(noisy_panel)
    estimated = [item for item in result.estimates if item.estimated]
    assert estimated
    assert result.n_obs == len(noisy_panel)
    assert result.n_regions == noisy_panel["region_id"].nunique()
    assert result.n_treated_regions == 20
    t_critical = float(stats.t.ppf(0.975, result.n_regions - 1))
    for item in estimated:
        assert item.estimate is not None
        assert item.std_error is not None and item.std_error > 0
        assert item.ci_low < item.estimate < item.ci_high
        assert 0 <= item.p_value <= 1
        assert item.ci_high == pytest.approx(item.estimate + t_critical * item.std_error)
        assert item.event_time != result.reference_event_time
    omitted_times = {item.event_time for item in result.omitted}
    assert result.reference_event_time in omitted_times


def test_pretrend_joint_test_uses_only_estimable_pre_periods(noisy_panel):
    result = fit_event_study(noisy_panel)
    joint = result.joint_pretrend
    assert joint.status == "success"
    assert joint.p_value is not None
    assert joint.df_numerator == len(joint.terms)
    assert joint.df_denominator == result.n_regions - 1
    by_term = {item.term: item for item in result.estimates}
    pre_estimated = [
        item.event_time
        for item in result.estimates
        if item.estimated and item.event_time < 0 and item.event_time != result.reference_event_time
    ]
    assert sorted(item.event_time for item in (by_term[term] for term in joint.terms)) == sorted(pre_estimated)
    assert result.reference_event_time not in pre_estimated
    assert all(event_time < 0 for event_time in pre_estimated)
    assert "不能" in joint.interpretation and "平行趋势" in joint.interpretation


def test_structural_event_study_recovers_static_effects(structural_panel):
    result = fit_event_study(structural_panel)
    for item in result.estimates:
        if item.role == "reference":
            assert item.estimate is None
        elif item.role == "pre":
            assert item.estimated
            assert item.estimate == pytest.approx(0.0, abs=1e-6)
        elif item.role == "post":
            assert item.estimate == pytest.approx(TARGET, abs=1e-6)


def test_event_study_does_not_modify_input_and_rejects_bad_samples(noisy_panel):
    snapshot = noisy_panel.copy(deep=True)
    fit_event_study(noisy_panel)
    pd.testing.assert_frame_equal(noisy_panel, snapshot)

    missing = copy_with_issue(noisy_panel, "missing_outcome")
    before = noisy_panel.copy(deep=True)
    with pytest.raises(ValueError, match="缺失值"):
        fit_event_study(missing)
    pd.testing.assert_frame_equal(noisy_panel, before)

    staggered = noisy_panel.copy(deep=True)
    region_id = staggered.loc[staggered["treated"] == 1, "region_id"].iloc[0]
    staggered.loc[staggered["region_id"] == region_id, "treatment_start_month"] = POLICY_START_MONTH - 2
    with pytest.raises(ValueError, match="统一处理时点"):
        fit_event_study(staggered)
    with pytest.raises(ValueError, match="样本为空"):
        fit_event_study(noisy_panel.iloc[0:0].copy())


def test_plot_uses_estimated_results_without_treating_reference_as_estimate(noisy_panel):
    result = fit_event_study(noisy_panel)
    ax = plot_event_study(result)
    containers = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
    assert len(containers) == 1
    plotted_x = list(containers[0].lines[0].get_xdata())
    plotted_y = list(containers[0].lines[0].get_ydata())
    estimated = [item for item in result.estimates if item.estimated]
    assert plotted_x == [item.event_time for item in estimated]
    assert plotted_y == pytest.approx([item.estimate for item in estimated])
    assert result.reference_event_time not in plotted_x
    assert any(abs(float(line.get_ydata()[0])) < 1e-12 for line in ax.lines if len(line.get_ydata()))
    assert any(np_is_close(line.get_xdata(), -0.5) for line in ax.lines)
    text = " ".join(text.get_text() for text in ax.figure.texts)
    assert "95%" in text
    assert "基准期" in text
    assert "平行趋势" in text
    assert ax.get_xlabel()
    assert ax.get_ylabel()
    plt.close(ax.figure)


def np_is_close(values, target: float) -> bool:
    return len(values) > 0 and all(abs(float(value) - target) < 1e-8 for value in values)
