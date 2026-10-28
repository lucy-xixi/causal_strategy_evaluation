"""按 region_type 分解处理效应，并检验组间差异。

分组估计回答某一类地区的效应是否远离零。交互项回答两类地区的效应是否不同。
这两个问题不能互相替代。模拟真值不进入估计函数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cluster_inference import (
    HypothesisTest,
    fit_clustered_formula,
    infer_common_policy_start,
    require_panel,
    wald_test_zero,
)
from src.did_analysis import fit_did

SMALL_ARM_BELOW = 8
EQUALITY_INTERPRETATION = (
    "联合检验的原假设是各 region_type 的处理效应彼此相等。"
    "它不检验各组效应是否为零。未拒绝相等，也不能证明各组效应真的相同；"
    "处理前趋势不显著同样不能证明平行趋势。"
)


@dataclass(frozen=True)
class GroupEffect:
    region_type: str
    status: str
    failure_reason: str | None
    estimate: float | None
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    p_value_bonferroni: float | None
    n_obs: int | None
    n_regions: int | None
    n_treated_regions: int | None
    n_control_regions: int | None
    warnings: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class Contrast:
    region_type: str
    reference_type: str
    status: str
    failure_reason: str | None
    difference: float | None
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    implied_effect: float | None
    hypothesis: str


@dataclass(frozen=True)
class HeterogeneityResult:
    groups: tuple[GroupEffect, ...]
    reference_type: str | None
    reference_effect: float | None
    contrasts: tuple[Contrast, ...]
    joint_equality: HypothesisTest
    multiple_testing_note: str
    warnings: tuple[str, ...]
    scope_note: str


def fit_heterogeneity(df: pd.DataFrame) -> HeterogeneityResult:
    """分组 DiD 和统一模型中的类型交互。不会修改 df。"""

    _require_inputs(df)
    types = tuple(dict.fromkeys(df["region_type"].astype(str)))
    raw_groups = tuple(_fit_group(df, region_type) for region_type in types)
    family_size = sum(group.status == "success" and group.p_value is not None for group in raw_groups)
    groups = tuple(_with_bonferroni(group, family_size) for group in raw_groups)
    reference = _choose_reference(groups)
    contrasts, reference_effect, joint = _fit_interactions(df, reference)
    return HeterogeneityResult(
        groups=groups,
        reference_type=reference,
        reference_effect=reference_effect,
        contrasts=contrasts,
        joint_equality=joint,
        multiple_testing_note=_multiple_testing_note(family_size),
        warnings=_summary_warnings(groups),
        scope_note=(
            "分组回归在该 region_type 内部使用地区固定效应、月份固定效应和按地区聚类的标准误。"
            "它估计的是这一类处理地区的平均效应，不是全样本的处理地区等权 ATT。"
            "交互模型用全部地区估计组间差异，聚类数是全部地区数；"
            "小组里很少的处理地区或对照地区仍然限制了相应差异的信息量。"
        ),
    )


def format_heterogeneity(result: HeterogeneityResult) -> str:
    lines = ["异质性分析（模拟数据）", result.scope_note, result.multiple_testing_note]
    for group in result.groups:
        if group.status != "success":
            lines.append(f"{group.region_type}: 失败。{group.failure_reason}")
            continue
        lines.append(
            f"{group.region_type}: 估计 {group.estimate:.4f}，95% 区间 "
            f"[{group.ci_low:.4f}, {group.ci_high:.4f}]，"
            f"处理地区 {group.n_treated_regions}，对照地区 {group.n_control_regions}，"
            f"行数 {group.n_obs}，聚类地区数 {group.n_regions}，未调整 p = {group.p_value:.4g}，"
            f"Bonferroni p = {group.p_value_bonferroni:.4g}。"
        )
        for warning in group.warnings:
            if "处理地区" in warning or "对照地区" in warning or "少于 30" in warning:
                lines.append(f"  警告：{warning}")
    if result.reference_type is not None:
        lines.append(f"交互模型参照组：{result.reference_type}，参照组效应 {result.reference_effect:.4f}。")
    for contrast in result.contrasts:
        if contrast.status != "success":
            lines.append(f"{contrast.region_type} 相对 {contrast.reference_type}: 失败。{contrast.failure_reason}")
            continue
        lines.append(
            f"{contrast.region_type} 减 {contrast.reference_type}: {contrast.difference:.4f}，"
            f"95% 区间 [{contrast.ci_low:.4f}, {contrast.ci_high:.4f}]，p = {contrast.p_value:.4g}。"
            f"隐含的该组效应 {contrast.implied_effect:.4f}。"
        )
    joint = result.joint_equality
    if joint.status == "success":
        lines.append(
            f"组间相等的联合检验：F({joint.df_numerator}, {joint.df_denominator}) = {joint.statistic:.4f}，"
            f"p = {joint.p_value:.4g}。{joint.null_hypothesis}"
        )
    else:
        lines.append(f"组间相等的联合检验未能执行：{joint.reason}")
    return "\n".join(lines)


def _fit_group(df: pd.DataFrame, region_type: str) -> GroupEffect:
    subset = df.loc[df["region_type"].astype(str) == region_type].copy(deep=True)
    note = (
        f"{region_type} 的估计来自该类型内部的双向固定效应 DiD。"
        "聚类单位是该类型的地区。这个系数不是全样本 ATT。"
    )
    try:
        fit = fit_did(subset)
    except (ValueError, RuntimeError) as exc:
        return GroupEffect(
            region_type=region_type,
            status="failed",
            failure_reason=str(exc),
            estimate=None,
            std_error=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
            p_value_bonferroni=None,
            n_obs=len(subset),
            n_regions=int(subset["region_id"].nunique()),
            n_treated_regions=None,
            n_control_regions=None,
            warnings=(),
            note=note,
        )
    warnings = _arm_warnings(fit.n_treated_regions, fit.n_control_regions, fit.n_regions)
    return GroupEffect(
        region_type=region_type,
        status="success",
        failure_reason=None,
        estimate=fit.estimate,
        std_error=fit.std_error,
        ci_low=fit.ci_low,
        ci_high=fit.ci_high,
        p_value=fit.p_value,
        p_value_bonferroni=None,
        n_obs=fit.n_obs,
        n_regions=fit.n_regions,
        n_treated_regions=fit.n_treated_regions,
        n_control_regions=fit.n_control_regions,
        warnings=tuple(warnings),
        note=note,
    )


def _fit_interactions(
    df: pd.DataFrame,
    reference: str | None,
) -> tuple[tuple[Contrast, ...], float | None, HypothesisTest]:
    if reference is None or not reference.isidentifier():
        failed = _failed_equality("没有可用于参照的地区类型。")
        return (), None, failed
    formula = (
        "outcome ~ treated_post + "
        f"treated_post:C(region_type, Treatment(reference='{reference}')) + "
        "C(region_id) + C(month)"
    )
    try:
        fit = fit_clustered_formula(df, formula, ("treated_post",))
    except (ValueError, RuntimeError) as exc:
        return (), None, _failed_equality(str(exc))
    reference_effect = fit.coefficient("treated_post").estimate
    interaction_names = _interaction_names(fit.exog_names, reference)
    if not interaction_names:
        return (), reference_effect, _failed_equality("模型没有返回地区类型交互项。")
    # 重新提取交互项系数。关注项在第一次调用里只有 treated_post，这里直接读拟合对象。
    coefficients = {
        term: _read_coefficient(fit, term) for term in ("treated_post", *interaction_names.values())
    }
    contrasts = []
    for region_type, term in interaction_names.items():
        coefficient = coefficients[term]
        contrasts.append(
            Contrast(
                region_type=region_type,
                reference_type=reference,
                status="success",
                failure_reason=None,
                difference=coefficient.estimate,
                std_error=coefficient.std_error,
                ci_low=coefficient.ci_low,
                ci_high=coefficient.ci_high,
                p_value=coefficient.p_value,
                implied_effect=reference_effect + coefficient.estimate,
                hypothesis=f"{region_type} 的处理效应减去参照组 {reference} 的处理效应等于零。",
            )
        )
    joint = wald_test_zero(
        fit.fitted,
        tuple(interaction_names.values()),
        n_regions=fit.n_regions,
        name="type_effect_equality",
        null_hypothesis="各 region_type 的处理效应与参照组相同，因此各类型效应彼此相等。",
        interpretation=EQUALITY_INTERPRETATION,
    )
    return tuple(contrasts), reference_effect, joint


def _read_coefficient(fit: object, term: str):
    from src.cluster_inference import _coefficient

    return _coefficient(fit.fitted, term, fit.t_critical, fit.t_df)


def _interaction_names(exog_names: tuple[str, ...], reference: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in exog_names:
        if "treated_post" not in name or "region_type" not in name:
            continue
        match = re.search(r"\[T\.(.+?)\]", name)
        if match is None:
            continue
        level = match.group(1)
        if level == reference:
            continue
        found[level] = name
    return found


def _choose_reference(groups: tuple[GroupEffect, ...]) -> str | None:
    usable = [
        group
        for group in groups
        if group.n_treated_regions is not None
        and group.n_control_regions is not None
        and group.n_treated_regions > 0
        and group.n_control_regions > 0
    ]
    if not usable:
        return None
    usable.sort(key=lambda group: (-min(group.n_treated_regions, group.n_control_regions), group.region_type))
    return usable[0].region_type


def _arm_warnings(n_treated: int, n_control: int, n_regions: int) -> list[str]:
    from src.cluster_inference import finite_cluster_warning

    warnings = list(finite_cluster_warning(n_regions))
    if n_treated < SMALL_ARM_BELOW:
        warnings.append(f"该组只有 {n_treated} 个处理地区，聚类推断所依赖的处理臂很小，标准误可能不稳定。")
    if n_control < SMALL_ARM_BELOW:
        warnings.append(f"该组只有 {n_control} 个对照地区，聚类推断所依赖的对照臂很小，标准误可能不稳定。")
    warnings.append("分组后可用于聚类的地区数减少，常规聚类标准误的有限聚类问题没有自动解决。")
    return warnings


def _summary_warnings(groups: tuple[GroupEffect, ...]) -> tuple[str, ...]:
    notes = []
    for group in groups:
        if group.n_treated_regions is not None and group.n_treated_regions < SMALL_ARM_BELOW:
            notes.append(f"{group.region_type} 只有 {group.n_treated_regions} 个处理地区。")
        if group.n_control_regions is not None and group.n_control_regions < SMALL_ARM_BELOW:
            notes.append(f"{group.region_type} 只有 {group.n_control_regions} 个对照地区。")
    notes.append("分组推断没有自动解决有限聚类问题，也不能只凭各组 p 值不同就认为组间效应不同。")
    return tuple(notes)


def _with_bonferroni(group: GroupEffect, family_size: int) -> GroupEffect:
    if group.status != "success" or group.p_value is None or family_size < 1:
        return group
    adjusted = min(1.0, family_size * float(group.p_value))
    return GroupEffect(**{**group.__dict__, "p_value_bonferroni": adjusted})


def _multiple_testing_note(family_size: int) -> str:
    return (
        f"共有 {family_size} 个成功的分组零效应检验。"
        "这些检验问的是某一组效应是否为零，不是两组效应是否不同。"
        "多次检验会增加偶然发现的机会。"
        f"Bonferroni 调整为 min(1, {family_size} × 未调整 p)，只覆盖这 {family_size} 个零效应检验，"
        "并且没有把检验之间的相关性考虑进去。"
        "各组显著性不同，不能直接说明组间效应不同。"
    )


def _failed_equality(reason: str) -> HypothesisTest:
    return HypothesisTest(
        name="type_effect_equality",
        null_hypothesis="各 region_type 的处理效应彼此相等。",
        status="failed",
        method="聚类稳健 Wald 检验。",
        statistic=None,
        df_numerator=None,
        df_denominator=None,
        p_value=None,
        terms=(),
        reason=reason,
        interpretation=EQUALITY_INTERPRETATION,
    )


def _require_inputs(df: pd.DataFrame) -> None:
    require_panel(
        df,
        ("region_id", "region_type", "month", "treated", "post", "treated_post", "outcome"),
    )
    infer_common_policy_start(df)
    treated = pd.to_numeric(df["treated"], errors="coerce")
    post = pd.to_numeric(df["post"], errors="coerce")
    interaction = pd.to_numeric(df["treated_post"], errors="coerce")
    if treated.isna().any() or not treated.isin([0, 1]).all():
        raise ValueError("treated 只能是 0 或 1。异质性估计不会改正非法取值。")
    if post.isna().any() or not post.isin([0, 1]).all():
        raise ValueError("post 只能是 0 或 1。异质性估计不会改正非法取值。")
    if not np.isclose(interaction.to_numpy(dtype=float), (treated * post).to_numpy(dtype=float)).all():
        raise ValueError("treated_post 与 treated * post 不一致。异质性估计不会重写处理变量。")
