"""统一处理时点下的事件研究。

所有处理地区必须在同一个月份开始接受处理。相对时间等于月份减去这个共同时点。
基准期默认是相对时间 -1，它被省略，用来消除事件时间虚拟变量与固定效应的共线。
被省略的基准期不是估计出来的零。本模块不估计交错采纳下的动态效应。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import qr

from src.cluster_inference import (
    HypothesisTest,
    finite_cluster_warning,
    fit_clustered_formula,
    infer_common_policy_start,
    region_arm_counts,
    require_panel,
    twoway_demean,
    wald_test_zero,
)

PRETREND_INTERPRETATION = (
    "处理前系数联合不显著，并不能证明平行趋势成立。"
    "该检验只评估当前可估计的处理前系数是否同时为零，"
    "没有纳入被省略的基准期，也不能证实潜在结果的平行趋势。"
)
SCOPE_NOTE = (
    "事件研究只适用于统一处理时点：所有处理地区在同一月份开始接受处理。"
    "它不是交错采纳情形下的完整动态效应估计。"
)


@dataclass(frozen=True)
class EventTimeEstimate:
    event_time: int
    term: str | None
    estimated: bool
    role: str
    estimate: float | None
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    note: str


@dataclass(frozen=True)
class EventStudyResult:
    estimates: tuple[EventTimeEstimate, ...]
    reference_event_time: int
    policy_start_month: int
    n_obs: int
    n_regions: int
    n_treated_regions: int
    n_control_regions: int
    joint_pretrend: HypothesisTest
    specification: str
    scope_note: str
    warnings: tuple[str, ...]
    ci_level: float

    @property
    def omitted(self) -> tuple[EventTimeEstimate, ...]:
        return tuple(item for item in self.estimates if not item.estimated)


def fit_event_study(
    df: pd.DataFrame,
    *,
    reference_event_time: int = -1,
) -> EventStudyResult:
    """估计统一时点事件研究。不会修改 df。"""

    require_panel(df, ("region_id", "month", "treated", "outcome"))
    _require_binary_treatment(df)
    if reference_event_time >= 0:
        raise ValueError("基准期必须是处理前的相对时间，不能使用实施当月或实施后的时期。")
    policy_start = infer_common_policy_start(df)
    work = df.copy(deep=True)
    month = pd.to_numeric(work["month"], errors="raise")
    if not np.array_equal(month.to_numpy(), np.floor(month.to_numpy())):
        raise ValueError("月份必须是整数。")
    work["event_time"] = month.astype(int) - policy_start
    event_times = tuple(sorted(int(value) for value in work["event_time"].unique()))
    if reference_event_time not in event_times:
        raise ValueError(f"基准期 {reference_event_time} 不在当前样本的相对时间中，无法省略它来识别模型。")
    n_treated, n_control = region_arm_counts(work)
    if n_treated == 0 or n_control == 0:
        raise ValueError("事件研究样本必须同时包含处理地区和对照地区。")

    candidate_terms = tuple(_term_name(event_time) for event_time in event_times if event_time != reference_event_time)
    for event_time, term in zip(
        [event_time for event_time in event_times if event_time != reference_event_time],
        candidate_terms,
        strict=True,
    ):
        work[term] = ((work["treated"].astype(int) == 1) & (work["event_time"] == event_time)).astype(float)

    kept, dropped = _screen_event_terms(work, candidate_terms)
    if not kept:
        raise ValueError("双向固定效应之后没有可估计的相对时期。")
    formula = "outcome ~ " + " + ".join(kept) + " + C(region_id) + C(month)"
    fit = fit_clustered_formula(work, formula, kept)
    by_term = {item.term: item for item in fit.coefficients}
    dropped_reason = dict(dropped)
    estimates = tuple(
        _event_row(event_time, reference_event_time, by_term, dropped_reason) for event_time in event_times
    )
    pre_terms = tuple(
        item.term
        for item in estimates
        if item.estimated and item.role == "pre" and item.term is not None
    )
    joint = wald_test_zero(
        fit.fitted,
        pre_terms,
        n_regions=fit.n_regions,
        name="pretrend_joint",
        null_hypothesis="所有可估计的处理前事件时间系数同时为零，不包含基准期。",
        interpretation=PRETREND_INTERPRETATION,
    )
    _validate_pretrend_terms(joint, estimates, reference_event_time)
    warnings = list(fit.warnings)
    warnings.append("地区和月份虚拟变量加上事件时间项之后，全部系数的聚类协方差可能不满秩；单个区间只使用对应系数的方差。")
    if dropped:
        warnings.append("部分相对时期在固定效应之后不可估计，这些时期没有被记为 0。")
    return EventStudyResult(
        estimates=estimates,
        reference_event_time=reference_event_time,
        policy_start_month=policy_start,
        n_obs=fit.n_obs,
        n_regions=fit.n_regions,
        n_treated_regions=n_treated,
        n_control_regions=n_control,
        joint_pretrend=joint,
        specification=formula,
        scope_note=SCOPE_NOTE,
        warnings=tuple(warnings),
        ci_level=fit.ci_level,
    )


def format_event_study(result: EventStudyResult) -> str:
    lines = [
        "事件研究（模拟数据，统一处理时点）",
        result.scope_note,
        f"真实共同实施月份 = {result.policy_start_month}，基准期 = {result.reference_event_time}（省略，不是估计值）。",
        (
            f"样本：{result.n_obs} 行，{result.n_regions} 个地区，"
            f"处理地区 {result.n_treated_regions}，对照地区 {result.n_control_regions}。"
        ),
        "相对时间  状态    系数      标准误     95%区间                         p值",
    ]
    for item in result.estimates:
        if item.estimated:
            lines.append(
                f"{item.event_time:>6}  已估计  {item.estimate:8.4f}  {item.std_error:8.4f}  "
                f"[{item.ci_low:8.4f}, {item.ci_high:8.4f}]  {item.p_value:.4g}"
            )
        else:
            lines.append(f"{item.event_time:>6}  未估计  {item.note}")
    joint = result.joint_pretrend
    if joint.status == "success":
        lines.append(
            "处理前联合检验："
            f"F({joint.df_numerator}, {joint.df_denominator}) = {joint.statistic:.4f}，"
            f"p = {joint.p_value:.4g}。时期 = { _pre_times(result) }。"
        )
    else:
        lines.append(f"处理前联合检验未能执行：{joint.reason}")
    lines.append(joint.interpretation)
    for warning in result.warnings:
        lines.append(f"注意：{warning}")
    return "\n".join(lines)


def _event_row(
    event_time: int,
    reference_event_time: int,
    by_term: dict[str, object],
    dropped_reason: dict[str, str],
) -> EventTimeEstimate:
    role = _role(event_time, reference_event_time)
    if event_time == reference_event_time:
        return EventTimeEstimate(
            event_time=event_time,
            term=None,
            estimated=False,
            role="reference",
            estimate=None,
            std_error=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
            note="基准期被省略，用来消除与固定效应的共线。这不是估计值为零。",
        )
    term = _term_name(event_time)
    if term in dropped_reason:
        return EventTimeEstimate(
            event_time,
            term,
            False,
            "not_estimable",
            None,
            None,
            None,
            None,
            None,
            dropped_reason[term],
        )
    coefficient = by_term[term]
    return EventTimeEstimate(
        event_time=event_time,
        term=term,
        estimated=True,
        role=role,
        estimate=coefficient.estimate,
        std_error=coefficient.std_error,
        ci_low=coefficient.ci_low,
        ci_high=coefficient.ci_high,
        p_value=coefficient.p_value,
        note="已估计的动态处理效应。",
    )


def _screen_event_terms(df: pd.DataFrame, terms: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    demeaned = twoway_demean(df, list(terms))
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for term in terms:
        if float(demeaned[term].std(ddof=0)) <= 1e-8:
            dropped.append((term, "双向去均值后没有剩余变异，该相对时期不可估计，未记为 0。"))
        else:
            kept.append(term)
    if not kept:
        return (), tuple(dropped)
    matrix = demeaned.loc[:, kept].to_numpy(dtype=float)
    independent = _independent_columns(matrix)
    kept_final = []
    for position, term in enumerate(kept):
        if position in independent:
            kept_final.append(term)
        else:
            dropped.append((term, "与其他事件时间项及固定效应共线，该相对时期不可估计，未记为 0。"))
    return tuple(kept_final), tuple(dropped)


def _independent_columns(matrix: np.ndarray) -> set[int]:
    _q, r_matrix, pivots = qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r_matrix))
    tolerance = 1e-8 * float(diagonal.max()) if diagonal.size else 0.0
    independent = {int(pivots[index]) for index, value in enumerate(diagonal) if value > tolerance}
    return independent


def _validate_pretrend_terms(
    joint: HypothesisTest,
    estimates: tuple[EventTimeEstimate, ...],
    reference_event_time: int,
) -> None:
    if joint.status != "success":
        return
    by_term = {item.term: item for item in estimates if item.term is not None}
    for term in joint.terms:
        item = by_term[term]
        if item.role != "pre" or not item.estimated or item.event_time == reference_event_time or item.event_time >= 0:
            raise RuntimeError("处理前联合检验纳入了基准期、处理后时期或未估计时期。")


def _pre_times(result: EventStudyResult) -> list[int]:
    by_term = {item.term: item.event_time for item in result.estimates if item.term is not None}
    return [by_term[term] for term in result.joint_pretrend.terms]


def _role(event_time: int, reference_event_time: int) -> str:
    if event_time == reference_event_time:
        return "reference"
    if event_time < 0:
        return "pre"
    return "post"


def _term_name(event_time: int) -> str:
    if event_time < 0:
        return f"et_m{abs(event_time)}"
    return f"et_{event_time}"


def _require_binary_treatment(df: pd.DataFrame) -> None:
    treated = pd.to_numeric(df["treated"], errors="coerce")
    if treated.isna().any() or not treated.isin([0, 1]).all():
        raise ValueError("treated 只能是 0 或 1。事件研究不会改正非法取值。")
