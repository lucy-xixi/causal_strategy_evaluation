"""策略评估的 Streamlit 页面。

页面调用数据生成、质量检查和已经实现的估计函数。
系数、区间和 p 值来自当次拟合，不在页面里写死。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DEFAULT_SEED, IDENTIFICATION_NOTES, tau_by_type
from src.data_generation import generate_panel
from src.data_quality import assess_panel
from src.did_analysis import fit_did
from src.event_study import fit_event_study
from src.heterogeneity import fit_heterogeneity
from src.plotting import plot_event_study, plot_interval_estimates, plot_outcome_trends, plot_placebo_distribution
from src.reporting import render_report, write_report
from src.robustness import run_robustness

PAGES = (
    "概览",
    "数据与质量",
    "分组趋势",
    "主要结果",
    "事件研究",
    "稳健性",
    "异质性",
    "决策报告",
)


@st.cache_data(show_spinner=False)
def load_analysis(seed: int, placebo_reps: int) -> dict[str, object]:
    """按种子生成一份面板并完成全部估计。缓存的是结果对象，不是写死的数字。"""

    panel = generate_panel(int(seed))
    return {
        "panel": panel,
        "quality": assess_panel(panel),
        "did": fit_did(panel, tau_by_type=tau_by_type()),
        "event": fit_event_study(panel),
        "robustness": run_robustness(panel, placebo_reps=int(placebo_reps), placebo_seed=int(seed)),
        "heterogeneity": fit_heterogeneity(panel),
    }


def main() -> None:
    st.set_page_config(page_title="策略评估平台", layout="wide")
    st.title("因果策略评估平台")
    seed, placebo_reps = _sidebar()
    try:
        with st.spinner("正在估计当前种子下的模型。虚假处理组会重复拟合，第一次会稍慢。"):
            bundle = load_analysis(seed, placebo_reps)
    except (ValueError, RuntimeError) as exc:
        st.error(f"当前种子没有完成估计：{exc}")
        return
    bundle = dict(bundle)
    bundle["seed"] = seed
    bundle["placebo_reps"] = placebo_reps
    st.info("这是固定种子生成的模拟面板，不是企业真实成效。一个 p 值不能单独决定是否扩大实施。")
    page = st.sidebar.radio("页面", PAGES)
    {
        "概览": _page_overview,
        "数据与质量": _page_quality,
        "分组趋势": _page_trends,
        "主要结果": _page_did,
        "事件研究": _page_event,
        "稳健性": _page_robustness,
        "异质性": _page_heterogeneity,
        "决策报告": _page_report,
    }[page](bundle)


def _sidebar() -> tuple[int, int]:
    st.sidebar.header("复现设置")
    seed = int(st.sidebar.number_input("数据种子", min_value=0, max_value=10_000, value=DEFAULT_SEED, step=1))
    placebo_reps = int(st.sidebar.slider("虚假处理组重复次数", min_value=5, max_value=30, value=30, step=1))
    st.sidebar.caption("虚假处理组使用与数据相同的种子。同一组种子和重复次数会重复使用缓存，更换后会重新估计。")
    return seed, placebo_reps


def _page_overview(bundle: dict[str, object]) -> None:
    did = bundle["did"]
    st.subheader("要回答的问题")
    st.markdown(
        """
这份模拟案例问的是：平台从某个月开始，在一部分地区实施补贴之后，单位用户收益是否变化；
这个变化有多少可以在双重差分的假设下写成策略效应；效应是否因地区类型而不同；
结论对设定和样本是否敏感；现有证据够不够支持下一步。
        """
    )
    left, right = st.columns(2)
    left.metric("基础 DiD 估计", f"{did.estimate:.3f}")
    right.metric("95% 区间", f"[{did.ci_low:.3f}, {did.ci_high:.3f}]")
    st.caption(
        f"样本量 {did.n_obs} 行，聚类地区数 {did.n_regions}，"
        f"处理地区 {did.n_treated_regions}，对照地区 {did.n_control_regions}。"
        "区间使用地区聚类方差和 t(G−1)。"
    )
    st.markdown(did.inference_note)
    with st.expander("模拟设计和识别说明"):
        st.markdown(IDENTIFICATION_NOTES)


def _page_quality(bundle: dict[str, object]) -> None:
    panel = bundle["panel"]
    quality = bundle["quality"]
    st.subheader("面板与质量检查")
    st.write(
        f"当前种子 {bundle['seed']} 生成 {len(panel)} 行、"
        f"{panel['region_id'].nunique()} 个地区、{panel['month'].nunique()} 个月。"
        f"日历从 {panel['year_month'].min()} 到 {panel['year_month'].max()}。"
    )
    rows = [
        {
            "检查": item.name,
            "级别": item.severity,
            "通过": item.passed,
            "问题数": item.n_issues,
            "说明": item.message,
        }
        for item in quality.checks
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("错误级别的检查决定整份报告是否通过。异常值诊断只提示，不自动删行。")
    summary = panel.groupby(["treated", "post"], as_index=False)["outcome"].agg(行数="size", 均值="mean", 标准差="std")
    st.markdown("**按处理身份和实施前后分组的结果均值**")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("这是原始均值。它还混有地区构成和共同季节性，不等于后面的双重差分估计。")


def _page_trends(bundle: dict[str, object]) -> None:
    panel = bundle["panel"]
    did = bundle["did"]
    st.subheader("处理组与对照组的平均轨迹")
    _pyplot(plot_outcome_trends(panel, did.policy_start_month))
    st.markdown(
        "两条线是各组在每个月的平均结果。竖线标在策略开始前半个月，避免把实施当月画成处理前。"
        "对照组的平均水平可以更高，因为处理名额在基线较低的地区类型里更多；"
        "基础模型用地区固定效应吸收这种水平差，用月份固定效应吸收共同季节性。"
        "处理前两条线大致平行，仍然不是平行趋势假设的证明。这张图也不是双重差分估计。"
    )


def _page_did(bundle: dict[str, object]) -> None:
    did = bundle["did"]
    st.subheader("基础双重差分")
    st.code(did.specification, language="text")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "估计": did.estimate,
                    "聚类标准误": did.std_error,
                    "区间下限": did.ci_low,
                    "区间上限": did.ci_high,
                    "p 值": did.p_value,
                    "t 自由度": did.t_df,
                    "临界值": did.t_critical,
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"{did.ci_level:.0%} 区间 [{did.ci_low:.3f}, {did.ci_high:.3f}]，临界值 {did.t_critical:.3f}。")
    _pyplot(
        plot_interval_estimates(
            ["基础 DiD"],
            [did.estimate],
            [did.ci_low],
            [did.ci_high],
            title="基础双重差分（模拟数据）",
            xlabel="处理地区平均效应",
        )
    )
    if did.target_effect is not None:
        comparison = (
            f"模拟设计中的处理地区等权目标是 {did.target_effect:.3f}。"
            + (f"本次估计与它相差 {did.bias:.3f}。" if did.bias is not None else "当前样本不满足等权解释，不计算偏差。")
        )
        st.markdown(comparison)
    st.markdown(did.inference_note)
    for warning in did.warnings:
        st.warning(warning)
    st.caption(
        f"估计使用 {did.n_obs} 行，删除 {did.n_dropped} 行；"
        f"月份 {did.month_min}–{did.month_max}，其中处理前 {did.n_pre_months} 个月、处理后 {did.n_post_months} 个月。"
    )


def _page_event(bundle: dict[str, object]) -> None:
    event = bundle["event"]
    st.subheader("事件研究")
    st.markdown(event.scope_note)
    _pyplot(plot_event_study(event))
    joint = event.joint_pretrend
    if joint.status == "success":
        st.markdown(
            f"处理前联合检验纳入 {joint.df_numerator} 个处理前系数，"
            f"F({joint.df_numerator}, {joint.df_denominator}) = {joint.statistic:.3f}，p = {joint.p_value:.4g}。"
        )
    else:
        st.warning(f"处理前联合检验没有给出统计量：{joint.reason}")
    st.markdown(joint.interpretation)
    rows = [
        {
            "相对时间": item.event_time,
            "角色": item.role,
            "估计": item.estimate,
            "标准误": item.std_error,
            "区间下限": item.ci_low,
            "区间上限": item.ci_high,
            "p 值": item.p_value,
            "说明": item.note,
        }
        for item in event.estimates
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"样本 {event.n_obs} 行，{event.n_regions} 个地区，处理地区 {event.n_treated_regions} 个。基准期没有估计值。")


def _page_robustness(bundle: dict[str, object]) -> None:
    suite = bundle["robustness"]
    st.subheader("稳健性")
    st.caption("显著性有没有变化，本身不是稳健性的结论。下面的数字来自各项检验自己的拟合。")
    frame = suite.to_frame()
    st.dataframe(frame, hide_index=True, width="stretch")
    plotted = [check for check in suite.checks if check.status == "success" and check.estimate is not None]
    if plotted:
        _pyplot(
            plot_interval_estimates(
                [check.name for check in plotted],
                [check.estimate for check in plotted],
                [check.ci_low for check in plotted],
                [check.ci_high for check in plotted],
                title="各项设定的估计和 95% 区间（模拟数据）",
                xlabel="双重差分估计",
            )
        )
    assignment = next((check for check in suite.checks if check.name == "placebo_assignment"), None)
    if assignment is not None and assignment.status == "success" and assignment.actual_estimate is not None:
        _pyplot(plot_placebo_distribution(assignment.placebo_estimates, assignment.actual_estimate))
        st.markdown(
            f"重复 {assignment.n_replications} 次，种子 {assignment.placebo_seed}。"
            f"虚假分配均值 {assignment.placebo_mean:.3f}，"
            f"真实估计 {assignment.actual_estimate:.3f} 的尾部比例是 {assignment.tail_fraction:.3f}。"
            f"{assignment.p_value_reason or ''}{assignment.limitations}"
        )
    for check in suite.checks:
        st.markdown(f"**{check.name}**")
        st.write(check.description)
        if check.status != "success":
            st.warning(check.failure_reason)
        else:
            st.write(check.interpretation)
            if check.comparison_to_baseline:
                st.write(check.comparison_to_baseline)
            st.caption(check.limitations)


def _page_heterogeneity(bundle: dict[str, object]) -> None:
    result = bundle["heterogeneity"]
    st.subheader("异质性")
    st.markdown(result.scope_note)
    st.markdown(result.multiple_testing_note)
    rows = []
    for group in result.groups:
        rows.append(
            {
                "地区类型": group.region_type,
                "状态": group.status,
                "估计": group.estimate,
                "区间下限": group.ci_low,
                "区间上限": group.ci_high,
                "未调整 p": group.p_value,
                "Bonferroni p": group.p_value_bonferroni,
                "处理地区": group.n_treated_regions,
                "对照地区": group.n_control_regions,
                "行数": group.n_obs,
                "聚类数": group.n_regions,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    plotted = [group for group in result.groups if group.status == "success"]
    if plotted:
        _pyplot(
            plot_interval_estimates(
                [group.region_type for group in plotted],
                [group.estimate for group in plotted],
                [group.ci_low for group in plotted],
                [group.ci_high for group in plotted],
                title="分类型双重差分（模拟数据）",
                xlabel="该类型处理地区的平均效应",
            )
        )
    for group in result.groups:
        for warning in group.warnings:
            st.warning(f"{group.region_type}：{warning}")
    if result.reference_type is not None:
        st.markdown(
            f"交互模型的参照组是 {result.reference_type}。"
            "选择规则是：在成功估计的类型中，取处理地区数和对照地区数的较小值最大者；并列时按类型名称排序。"
            f"参照组系数是 {result.reference_effect:.3f}。交互项是其他类型相对参照组的差异，不是该类型效应是否为零。"
        )
    contrast_rows = [
        {
            "比较": f"{item.region_type} − {item.reference_type}",
            "差异": item.difference,
            "区间下限": item.ci_low,
            "区间上限": item.ci_high,
            "p 值": item.p_value,
            "隐含该组效应": item.implied_effect,
        }
        for item in result.contrasts
        if item.status == "success"
    ]
    if contrast_rows:
        st.dataframe(pd.DataFrame(contrast_rows), hide_index=True, width="stretch")
    joint = result.joint_equality
    if joint.status == "success":
        st.markdown(
            f"组间相等检验：F({joint.df_numerator}, {joint.df_denominator}) = {joint.statistic:.3f}，"
            f"p = {joint.p_value:.4g}。原假设：{joint.null_hypothesis}"
        )
    else:
        st.warning(f"组间相等检验没有给出统计量：{joint.reason}")
    st.markdown(joint.interpretation)


def _page_report(bundle: dict[str, object]) -> None:
    st.subheader("自动报告")
    preamble = (
        f"数据种子 {bundle['seed']}，虚假处理组重复 {bundle['placebo_reps']} 次、种子 {bundle['seed']}。"
        "以下系数、区间、样本量和 p 值来自本次拟合，不是页面中保存的示例数字。"
    )
    text = render_report(
        bundle["panel"],
        bundle["quality"],
        bundle["did"],
        bundle["event"],
        bundle["robustness"],
        bundle["heterogeneity"],
        preamble=preamble,
    )
    st.markdown(text)
    filename = f"strategy_evaluation_seed{bundle['seed']}_reps{bundle['placebo_reps']}.md"
    st.download_button("下载 Markdown", text, file_name=filename, mime="text/markdown")
    if st.button("保存到 reports/"):
        path = write_report(text, Path("reports") / filename)
        st.success(f"已保存到 {path.as_posix()}")


def _pyplot(ax) -> None:
    st.pyplot(ax.figure, clear_figure=True)


if __name__ == "__main__":
    main()
