import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal

from src.config import tau_by_type
from src.data_generation import generate_panel
from src.data_quality import assess_panel
from src.did_analysis import fit_did
from src.event_study import fit_event_study
from src.heterogeneity import fit_heterogeneity
from src.plotting import plot_interval_estimates, plot_outcome_trends
from src.reporting import render_report, write_report
from src.robustness import run_robustness


def test_report_uses_fitted_results_and_keeps_placebo_time_before_policy(tmp_path):
    panel = generate_panel(7)
    original = panel.copy()
    quality = assess_panel(panel)
    did = fit_did(panel, tau_by_type=tau_by_type())
    event = fit_event_study(panel)
    robustness = run_robustness(panel, placebo_reps=2, placebo_seed=7)
    heterogeneity = fit_heterogeneity(panel)
    text = render_report(panel, quality, did, event, robustness, heterogeneity)
    assert_frame_equal(panel, original)

    assert f"{did.estimate:.3f}" in text
    assert f"{did.ci_low:.3f}" in text
    assert f"{did.ci_high:.3f}" in text
    assert f"{did.p_value:.4g}" in text
    assert str(did.n_obs) in text
    assert str(did.n_treated_regions) in text
    for heading in ("业务问题", "数据来源", "数据质量", "分析方法", "主要效果", "平行趋势", "异质性", "局限"):
        assert heading in text
    assert "模拟" in text
    assert "不能根据 p 值自动决定扩大实施" in text
    assert "全面推广" not in text

    placebo = next(check for check in robustness.checks if check.name == "placebo_time")
    assert placebo.status == "success"
    assert placebo.month_max < did.policy_start_month
    assert f"{placebo.month_min}–{placebo.month_max}" in text
    assignment = next(check for check in robustness.checks if check.name == "placebo_assignment")
    assert assignment.p_value is None
    assert "不是识别假设的 p 值" in text
    assert f"{heterogeneity.reference_type}" in text

    if event.joint_pretrend.status == "success":
        assert f"{event.joint_pretrend.statistic:.3f}" in text
        assert event.joint_pretrend.df_numerator == len(event.joint_pretrend.terms)

    path = write_report(text, tmp_path / "report.md")
    assert path.read_text(encoding="utf-8") == text


def test_interval_and_trend_plots_use_the_supplied_values():
    panel = generate_panel(1)
    original = panel.copy()
    trends = plot_outcome_trends(panel, 12)
    assert_frame_equal(panel, original)
    vertical = [
        float(line.get_xdata()[0])
        for line in trends.get_lines()
        if len(line.get_xdata()) >= 2 and line.get_xdata()[0] == line.get_xdata()[-1]
    ]
    assert 11.5 in vertical
    assert trends.get_ylabel() == "平均单位用户收益"
    plt.close(trends.figure)

    intervals = plot_interval_estimates(
        ["甲", "乙"],
        [1.25, -0.5],
        [0.25, -1.5],
        [2.25, 0.5],
        title="区间",
        xlabel="效应",
    )
    assert list(intervals.get_lines()[0].get_xdata()) == [1.25, -0.5]
    assert intervals.get_xlabel() == "效应"
    plt.close(intervals.figure)


def test_app_lists_the_eight_pages():
    import app

    assert app.PAGES == ("概览", "数据与质量", "分组趋势", "主要结果", "事件研究", "稳健性", "异质性", "决策报告")
    assert callable(app.main)
    assert callable(app.load_analysis)
