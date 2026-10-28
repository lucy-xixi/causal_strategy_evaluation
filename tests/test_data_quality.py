import pandas as pd
import pytest

from src.data_generation import generate_panel
from src.data_quality import assess_panel, copy_with_issue


@pytest.fixture
def panel() -> pd.DataFrame:
    return generate_panel(4)


def test_clean_panel_passes_error_checks(panel):
    report = assess_panel(panel)
    assert report.passed
    for name in (
        "missing_values",
        "duplicate_region_month",
        "time_range",
        "panel_completeness",
        "group_distribution",
        "treatment_logic",
    ):
        assert report.check(name).passed, report.check(name).message
    assert report.check("treatment_logic").details["rules"] == []
    distribution = report.check("group_distribution").details
    assert distribution["n_treated_regions"] == 20
    assert distribution["n_control_regions"] == 28
    assert report.check("outlier_diagnostic").severity == "info"


@pytest.mark.parametrize(
    ("issue", "check_name", "rule"),
    [
        ("duplicate_region_month", "duplicate_region_month", None),
        ("missing_outcome", "missing_values", None),
        ("illegal_treated_code", "treatment_logic", "treated_code"),
        ("illegal_interaction", "treatment_logic", "interaction"),
        ("illegal_post_timing", "treatment_logic", "post_timing"),
        ("month_out_of_range", "time_range", None),
        ("missing_region_month", "panel_completeness", None),
        ("treated_start_misaligned", "treatment_logic", "treated_start"),
    ],
)
def test_known_issues_are_detected_on_copies(panel, issue, check_name, rule):
    snapshot = panel.copy(deep=True)
    bad = copy_with_issue(panel, issue)
    assert bad is not panel
    pd.testing.assert_frame_equal(panel, snapshot)

    before = bad.copy(deep=True)
    report = assess_panel(bad)
    pd.testing.assert_frame_equal(bad, before)
    assert not report.passed or check_name == "outlier_diagnostic"
    found = report.check(check_name)
    assert not found.passed
    assert found.n_issues > 0
    if rule is not None:
        assert rule in found.details["rules"]


def test_outlier_injection_is_diagnostic_and_does_not_change_source(panel):
    snapshot = panel.copy(deep=True)
    bad = panel.copy(deep=True)
    bad.loc[bad.index[0], "outcome"] = 1_000_000.0
    report = assess_panel(bad)
    pd.testing.assert_frame_equal(panel, snapshot)
    assert report.passed
    outlier = report.check("outlier_diagnostic")
    assert not outlier.passed
    assert outlier.n_issues >= 1
    assert "不会删除" in outlier.message


def test_control_start_month_is_allowed_to_be_missing(panel):
    missing = assess_panel(panel).check("missing_values")
    assert missing.passed
    assert "treatment_start_month" not in missing.details["na_counts"]
    assert panel["treatment_start_month"].isna().any()
