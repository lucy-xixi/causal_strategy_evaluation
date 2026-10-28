"""把已经算好的分析结果写成 Markdown 报告。

报告区分数据事实、模型估计、依赖假设的解释和尚未确定的事项。
它不根据 p 值自动给出全面推广的结论。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import IDENTIFICATION_NOTES, REGION_TYPES, tau_by_type
from src.data_quality import QualityReport, assess_panel
from src.did_analysis import DIDResult, fit_did
from src.estimand import design_target
from src.event_study import EventStudyResult, fit_event_study
from src.heterogeneity import HeterogeneityResult, fit_heterogeneity
from src.robustness import RobustnessSuite, run_robustness

POLICY_CHECK_NAMES = (
    "covariate_promo_intensity",
    "window_pre6_post6",
    "window_pre3_post3",
)


def build_report(
    panel: pd.DataFrame,
    *,
    placebo_reps: int = 30,
    placebo_seed: int = 42,
) -> str:
    """从面板重新估计并生成报告。不修改 panel。"""

    quality = assess_panel(panel)
    did = fit_did(panel, tau_by_type=tau_by_type())
    event = fit_event_study(panel)
    robustness = run_robustness(panel, placebo_reps=placebo_reps, placebo_seed=placebo_seed)
    heterogeneity = fit_heterogeneity(panel)
    return render_report(
        panel,
        quality,
        did,
        event,
        robustness,
        heterogeneity,
        preamble=(
            f"虚假处理组重复 {placebo_reps} 次，随机种子 {placebo_seed}。"
            "报告中的系数、区间、样本量和 p 值来自本次拟合。"
        ),
    )


def render_report(
    panel: pd.DataFrame,
    quality: QualityReport,
    did: DIDResult,
    event: EventStudyResult,
    robustness: RobustnessSuite,
    heterogeneity: HeterogeneityResult,
    *,
    preamble: str = "",
) -> str:
    """只用调用方传入的结果写报告，不另外硬编码系数。"""

    outcome_mean = float(panel["outcome"].mean())
    sections = []
    if preamble.strip():
        sections.append(preamble.strip())
    sections.extend([
        _business_question(),
        _data_section(panel, quality),
        _method_section(did),
        _main_effect_section(did, outcome_mean),
        _diagnostics_section(event, robustness),
        _heterogeneity_section(heterogeneity),
        _decision_section(did, event, robustness, heterogeneity, outcome_mean),
    ])
    return "\n\n".join(sections) + "\n"


def write_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _business_question() -> str:
    return "\n".join(
        [
            "# 运营补贴策略评估报告",
            "",
            "本报告使用模拟数据，展示一次地区层面补贴策略的评估过程。它不是企业真实成效。",
            "",
            "## 1. 业务问题",
            "",
            "模拟情景是：某运营平台从某个月开始，在部分地区实施补贴，希望提高单位用户收益。其他地区尚未实施。",
            "",
            "需要分开回答五件事：",
            "",
            "1. 实施之后，单位用户收益的样本均值是否变化。",
            "2. 在地区固定效应、月份固定效应和平行趋势假设下，变化有多少可以写成策略的平均效应。",
            "3. 不同地区类型的估计是否不同。",
            "4. 估计对协变量、时间窗口和安慰剂设定是否敏感。",
            "5. 现有证据能支持什么样的下一步，而不是只看一个 p 值。",
        ]
    )


def _data_section(panel: pd.DataFrame, quality: QualityReport) -> str:
    treated = int(panel.drop_duplicates("region_id")["treated"].sum())
    regions = int(panel["region_id"].nunique())
    lines = [
        "## 2. 数据来源与模拟设定",
        "",
        "数据由固定种子的生成器产生。处理地区等权的设计目标是 "
        f"{design_target(REGION_TYPES):.1f}。这个数字来自定义，不是从回归里反推出来的企业事实。",
        "",
        IDENTIFICATION_NOTES,
        "",
        "### 数据质量",
        "",
        f"当前面板有 {len(panel)} 行、{regions} 个地区，其中处理地区 {treated} 个。"
        f"错误级别检查{'全部通过' if quality.passed else '存在未通过项'}。",
        "",
    ]
    for check in quality.checks:
        status = "通过" if check.passed else "未通过"
        lines.append(
            f"- {check.name}（{check.severity}）：{status}，问题数 {check.n_issues}。{check.message}"
        )
    lines.append("")
    lines.append("质量检查只报告问题，不删除或改写原始模拟数据。异常值诊断即使标记观测，也不进入自动删样。")
    return "\n".join(lines)


def _method_section(did: DIDResult) -> str:
    return "\n".join(
        [
            "## 3. 分析方法与识别假设",
            "",
            f"基础模型是 `{did.specification}`。",
            "地区固定效应吸收不随时间变化的地区差异，月份固定效应吸收共同趋势、季节性和共同冲击。",
            "在平衡面板且所有处理地区同时开始时，`treated_post` 的系数对应处理地区等权平均效应。",
            "它不是三种地区类型效应的简单平均，也不是按全部地区数加权的平均。",
            "",
            did.inference_note,
            "",
            "识别还依赖这些没有被统计量自动证明的假设：处理前没有系统性的组间差异趋势，"
            "没有预期效应和地区间溢出，补贴不影响用来做控制的变量。",
            "`promo_intensity` 在设计中不受策略影响，只在稳健性模型里加入。`region_baseline` 被地区固定效应吸收。",
        ]
    )


def _main_effect_section(did: DIDResult, outcome_mean: float) -> str:
    relative = did.estimate / outcome_mean if outcome_mean else float("nan")
    lines = [
        "## 4. 主要效果估计",
        "",
        "下面的数字来自当前样本上的基础双重差分，不是预先写好的示例。",
        "",
        f"- 估计值：{did.estimate:.3f}",
        f"- 聚类标准误：{did.std_error:.3f}",
        f"- {did.ci_level:.0%} 区间：[{did.ci_low:.3f}, {did.ci_high:.3f}]",
        f"- p 值：{did.p_value:.4g}",
        f"- 样本量：{did.n_obs} 行，{did.n_regions} 个地区，处理地区 {did.n_treated_regions} 个，对照地区 {did.n_control_regions} 个",
        f"- 月份范围：{did.month_min}–{did.month_max}，处理前 {did.n_pre_months} 个月，处理后 {did.n_post_months} 个月",
        f"- 拟合时删除的观测：{did.n_dropped}",
        "",
        f"样本平均单位用户收益是 {outcome_mean:.3f}。点估计约相当于这个平均水平的 {relative:.1%}。",
        "这是幅度描述。区间是否包含零，回答的是统计变异；它不等于这个幅度已经覆盖实施成本。",
    ]
    if did.target_effect is not None and did.bias is not None:
        lines.append(
            f"因为数据是模拟的，可以把估计值 {did.estimate:.3f} 和设计目标 {did.target_effect:.3f} 比较，"
            f"差值是 {did.bias:.3f}。"
        )
        lines.append("真实业务数据没有这种已知目标，不能用同一种偏差检查代替识别假设。")
    elif did.target_effect is not None:
        lines.append(
            f"设计目标是 {did.target_effect:.3f}，但当前样本不满足处理地区等权的解释条件，所以不计算偏差。"
        )
    for warning in did.warnings:
        lines.append(f"注意：{warning}")
    return "\n".join(lines)


def _diagnostics_section(event: EventStudyResult, robustness: RobustnessSuite) -> str:
    joint = event.joint_pretrend
    lines = [
        "## 5. 平行趋势和稳健性",
        "",
        event.scope_note,
        f"事件研究样本为 {event.n_obs} 行、{event.n_regions} 个地区、处理地区 {event.n_treated_regions} 个。",
        f"基准期是相对时间 {event.reference_event_time}，该期被省略，不是估计出来的零。",
        "",
    ]
    if joint.status == "success":
        lines.append(
            f"处理前联合检验使用 {joint.df_numerator} 个可估计的处理前系数，"
            f"F({joint.df_numerator}, {joint.df_denominator}) = {joint.statistic:.3f}，p = {joint.p_value:.4g}。"
        )
    else:
        lines.append(f"处理前联合检验没有给出统计量。原因：{joint.reason}")
    lines.append(joint.interpretation)
    lines.append("")
    lines.append("稳健性设定的估计如下。安慰剂时间使用的是真实实施前的样本；虚假处理组给出的是重新分配后的分布。")
    lines.append("")
    for check in robustness.checks:
        lines.append(_format_check(check))
    return "\n".join(lines)


def _heterogeneity_section(result: HeterogeneityResult) -> str:
    lines = [
        "## 6. 异质性",
        "",
        result.scope_note,
        "",
        result.multiple_testing_note,
        "",
    ]
    for group in result.groups:
        if group.status != "success":
            lines.append(f"- {group.region_type}：未能估计。{group.failure_reason}")
            continue
        lines.append(
            f"- {group.region_type}：估计 {group.estimate:.3f}，"
            f"区间 [{group.ci_low:.3f}, {group.ci_high:.3f}]，"
            f"未调整 p = {group.p_value:.4g}，Bonferroni p = {group.p_value_bonferroni:.4g}，"
            f"处理地区 {group.n_treated_regions}，对照地区 {group.n_control_regions}，"
            f"{group.n_obs} 行，聚类数 {group.n_regions}。"
        )
        for warning in group.warnings:
            lines.append(f"  - {warning}")
    if result.reference_type is not None and result.reference_effect is not None:
        lines.append("")
        lines.append(
            f"交互模型以 {result.reference_type} 为参照，参照组系数是 {result.reference_effect:.3f}。"
            "下面的差异是“该组减去参照组”，不是该组效应是否为零。"
        )
    for contrast in result.contrasts:
        if contrast.status != "success":
            lines.append(f"- {contrast.region_type} 相对 {contrast.reference_type}：未能估计。{contrast.failure_reason}")
            continue
        lines.append(
            f"- {contrast.region_type} 减 {contrast.reference_type}：{contrast.difference:.3f}，"
            f"区间 [{contrast.ci_low:.3f}, {contrast.ci_high:.3f}]，p = {contrast.p_value:.4g}。"
        )
    joint = result.joint_equality
    if joint.status == "success":
        lines.append(
            f"组间相等的联合检验：F({joint.df_numerator}, {joint.df_denominator}) = {joint.statistic:.3f}，"
            f"p = {joint.p_value:.4g}。{joint.null_hypothesis}"
        )
    else:
        lines.append(f"组间相等的联合检验没有给出统计量。原因：{joint.reason}")
    lines.append(joint.interpretation)
    return "\n".join(lines)


def _decision_section(
    did: DIDResult,
    event: EventStudyResult,
    robustness: RobustnessSuite,
    heterogeneity: HeterogeneityResult,
    outcome_mean: float,
) -> str:
    policy_checks = [
        check
        for check in robustness.checks
        if check.name in POLICY_CHECK_NAMES and check.status == "success" and check.estimate is not None
    ]
    same_direction = all(np.sign(check.estimate) == np.sign(did.estimate) for check in policy_checks)
    interval_excludes_zero = did.ci_low > 0 or did.ci_high < 0
    placebo = next((check for check in robustness.checks if check.name == "placebo_time"), None)
    assignment = next((check for check in robustness.checks if check.name == "placebo_assignment"), None)
    lines = [
        "## 7. 结论、局限与下一步",
        "",
        "### 数据直接显示的事实",
        "",
        f"面板有 {did.n_obs} 个地区—月份观测。样本平均结果是 {outcome_mean:.3f}。",
        "",
        "### 模型估计",
        "",
        f"基础双重差分估计是 {did.estimate:.3f}，{did.ci_level:.0%} 区间是 [{did.ci_low:.3f}, {did.ci_high:.3f}]。",
        _policy_comparison_sentence(policy_checks, same_direction),
        _placebo_sentence(placebo, assignment),
        "",
        "### 依赖假设的解释",
        "",
        "若平行趋势、无预期、无溢出，并且处理确实在统一月份实施，这个系数可以解释为处理地区的平均补贴效应。",
        event.joint_pretrend.interpretation,
        "",
        "### 尚不能确定的事项",
        "",
        "实施成本、用户间溢出、预期行为、以及结果能否推广到其他时期或其他平台，都不在当前数据里。",
        "常规聚类标准误在有限个地区下可能偏小，报告没有使用 CR2 或 wild cluster bootstrap。",
        heterogeneity.multiple_testing_note,
        "",
        "### 建议",
        "",
        _recommendation(interval_excludes_zero, same_direction, did),
    ]
    return "\n".join(lines)


def _policy_comparison_sentence(checks: list, same_direction: bool) -> str:
    if not checks:
        return "协变量或缩短窗口的设定没有成功估计，因此不能从替代设定判断方向是否稳定。"
    names = "、".join(check.name for check in checks)
    if same_direction:
        return f"{names} 的点估计与基准模型方向相同。方向相同仍然要同时看区间宽度和样本量。"
    return f"{names} 中有的点估计与基准模型方向不同。这时不能把基准模型的显著性单独当成稳定结论。"


def _placebo_sentence(placebo, assignment) -> str:
    parts = []
    if placebo is not None and placebo.status == "success" and placebo.estimate is not None:
        parts.append(
            f"安慰剂时间估计是 {placebo.estimate:.3f}，样本月份为 {placebo.month_min}–{placebo.month_max}，"
            "不含真实实施月及之后。它接近零只说明这个虚假时点没有再现基准效应，不能证明平行趋势。"
        )
    elif placebo is not None:
        parts.append(f"安慰剂时间没有成功估计。{placebo.failure_reason}")
    if assignment is not None and assignment.status == "success" and assignment.actual_estimate is not None:
        parts.append(
            f"虚假处理组重复 {assignment.n_replications} 次，种子 {assignment.placebo_seed}，"
            f"分布均值 {assignment.placebo_mean:.3f}。"
            f"真实估计 {assignment.actual_estimate:.3f} 的尾部比例是 {assignment.tail_fraction:.3f}。"
            "这是模拟分布中的位置，不是识别假设的 p 值。"
        )
    return "".join(parts)


def _recommendation(interval_excludes_zero: bool, same_direction: bool, did: DIDResult) -> str:
    opening = (
        f"建议同时看点估计 {did.estimate:.3f}、区间 [{did.ci_low:.3f}, {did.ci_high:.3f}]、"
        "替代设定、分组样本和尚未观测到的成本。"
    )
    if interval_excludes_zero and same_direction:
        decision = (
            "在这份模拟案例中，基准区间不包含零，而且已成功的协变量与缩短窗口设定方向一致。"
            "更稳妥的下一步是在相似地区做范围有限的后续试验，并事先写明继续或停止的规则。"
        )
    else:
        decision = (
            "基准区间或替代设定没有给出稳定、远离零的证据。"
            "这份模拟结果不足以支持扩大实施，下一步应先补充设计更清楚的试验。"
        )
    return opening + decision + "不能根据 p 值自动决定扩大实施，也不能把模拟结果写成真实企业成效。"


def _format_check(check) -> str:
    if check.status != "success":
        return f"- {check.name}：未能估计。{check.failure_reason}"
    if check.estimate is None:
        return (
            f"- {check.name}：重复 {check.n_replications} 次，种子 {check.placebo_seed}，"
            f"均值 {check.placebo_mean:.3f}，中位数 {check.placebo_p50:.3f}，"
            f"5% 到 95% 为 [{check.placebo_p05:.3f}, {check.placebo_p95:.3f}]，"
            f"真实估计 {check.actual_estimate:.3f}，尾部比例 {check.tail_fraction:.3f}。"
            f"{check.interpretation}{check.comparison_to_baseline}{check.limitations}"
        )
    return (
        f"- {check.name}：估计 {check.estimate:.3f}，区间 [{check.ci_low:.3f}, {check.ci_high:.3f}]，"
        f"p = {check.p_value:.4g}，{check.n_obs} 行，{check.n_regions} 个地区，"
        f"处理地区 {check.n_treated_regions} 个，月份 {check.month_min}–{check.month_max}。"
        f"{check.interpretation}{check.limitations}"
    )
