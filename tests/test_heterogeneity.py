import pandas as pd
import pytest

from src.config import REGION_TYPES, tau_by_type
from src.data_generation import generate_panel
from src.data_quality import copy_with_issue
from src.heterogeneity import SMALL_ARM_BELOW, fit_heterogeneity


def _expected_reference() -> str:
    ranked = sorted(
        REGION_TYPES,
        key=lambda spec: (-min(spec.n_treated, spec.n_regions - spec.n_treated), spec.name),
    )
    return ranked[0].name


@pytest.fixture(scope="module")
def structural_result():
    return fit_heterogeneity(generate_panel(5, structural=True))


@pytest.fixture(scope="module")
def noisy_result():
    return fit_heterogeneity(generate_panel(5))


def test_group_counts_match_the_design(noisy_result):
    by_type = {group.region_type: group for group in noisy_result.groups}
    assert set(by_type) == {spec.name for spec in REGION_TYPES}
    for spec in REGION_TYPES:
        group = by_type[spec.name]
        assert group.status == "success"
        assert group.n_regions == spec.n_regions
        assert group.n_treated_regions == spec.n_treated
        assert group.n_control_regions == spec.n_regions - spec.n_treated
        assert group.n_obs == spec.n_regions * 24
        assert group.estimate is not None
        assert group.ci_low < group.estimate < group.ci_high
        assert group.p_value_bonferroni == pytest.approx(min(1.0, 3 * group.p_value))


def test_small_sample_warnings_follow_the_arm_sizes(noisy_result):
    by_type = {group.region_type: group for group in noisy_result.groups}
    core = by_type["core"]
    emerging = by_type["emerging"]
    assert core.n_treated_regions < SMALL_ARM_BELOW
    assert emerging.n_control_regions < SMALL_ARM_BELOW
    assert any(
        "处理地区" in warning and str(core.n_treated_regions) in warning for warning in core.warnings
    )
    assert any(
        "对照地区" in warning and str(emerging.n_control_regions) in warning
        for warning in emerging.warnings
    )
    assert any("少于 30" in warning for warning in core.warnings)
    assert any("没有自动解决" in warning for warning in core.warnings)
    assert "不能" in noisy_result.multiple_testing_note
    assert "Bonferroni" in noisy_result.multiple_testing_note
    joined = " ".join(noisy_result.warnings)
    assert "core" in joined and "emerging" in joined


def test_structural_groups_and_contrasts_recover_type_effects(structural_result):
    taus = tau_by_type()
    reference = _expected_reference()
    assert structural_result.reference_type == reference
    assert structural_result.reference_effect == pytest.approx(taus[reference], abs=1e-6)
    for group in structural_result.groups:
        assert group.estimate == pytest.approx(taus[group.region_type], abs=1e-6)
    assert structural_result.contrasts
    for contrast in structural_result.contrasts:
        assert contrast.difference == pytest.approx(
            taus[contrast.region_type] - taus[contrast.reference_type],
            abs=1e-6,
        )
        assert contrast.implied_effect == pytest.approx(taus[contrast.region_type], abs=1e-6)
        assert contrast.hypothesis
    assert "相等" in structural_result.joint_equality.null_hypothesis
    assert "为零" not in structural_result.joint_equality.null_hypothesis or "不是" in (
        structural_result.joint_equality.interpretation
    )


def test_heterogeneity_is_reproducible_and_does_not_modify_data():
    panel = generate_panel(9)
    snapshot = panel.copy(deep=True)
    first = fit_heterogeneity(panel)
    second = fit_heterogeneity(panel)
    pd.testing.assert_frame_equal(panel, snapshot)
    assert [group.estimate for group in first.groups] == [group.estimate for group in second.groups]


def test_illegal_and_unidentifiable_group_are_explicit():
    panel = generate_panel(9)
    snapshot = panel.copy(deep=True)
    missing = copy_with_issue(panel, "missing_outcome")
    with pytest.raises(ValueError, match="缺失值"):
        fit_heterogeneity(missing)
    illegal = copy_with_issue(panel, "illegal_interaction")
    with pytest.raises(ValueError, match="treated_post"):
        fit_heterogeneity(illegal)
    pd.testing.assert_frame_equal(panel, snapshot)

    emerging_only = panel.copy(deep=True)
    mask = (emerging_only["region_type"] == "emerging") & (emerging_only["treated"] == 0)
    emerging_only.loc[mask, "treated"] = 1
    emerging_only.loc[mask, "treated_post"] = emerging_only.loc[mask, "post"].astype(int)
    emerging_only.loc[mask, "treatment_start_month"] = 12
    result = fit_heterogeneity(emerging_only)
    emerging = next(group for group in result.groups if group.region_type == "emerging")
    assert emerging.status == "failed"
    assert emerging.estimate is None
    assert emerging.failure_reason
    core = next(group for group in result.groups if group.region_type == "core")
    assert core.status == "success"
    pd.testing.assert_frame_equal(panel, snapshot)
