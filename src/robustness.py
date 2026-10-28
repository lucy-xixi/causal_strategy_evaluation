"""稳健性检验。

基准模型仍调用 fit_did。协变量、时间窗口、安慰剂时间和虚假处理组都返回结构化结果。
显著性是否改变，本身不构成稳健性结论。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cluster_inference import (
    finite_cluster_warning,
    fit_clustered_formula,
    infer_common_policy_start,
    region_arm_counts,
    require_panel,
    twoway_demean,
)
from src.did_analysis import DIDResult, fit_did

DEFAULT_PLACEBO_REPS = 30
DEFAULT_PLACEBO_SEED = 42
DEFAULT_WINDOW_OFFSETS = ((6, 6), (3, 3))
ASSIGNMENT_RULE = (
    "在每个 region_type 内，无放回抽取与该类型当前处理地区数相同的虚假处理地区；"
    "因此虚假处理组的规模和类型构成与当前样本一致。保留原来的 post 和统一实施月份。"
    "这不是真实随机实验，也不能证明平行趋势。"
)


@dataclass(frozen=True)
class RobustnessCheck:
    name: str
    description: str
    status: str
    failure_reason: str | None
    estimate: float | None
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    p_value_reason: str | None
    n_obs: int | None
    n_regions: int | None
    n_treated_regions: int | None
    n_control_regions: int | None
    month_min: int | None
    month_max: int | None
    policy_start_month: int | None
    placebo_policy_start: int | None
    window_note: str
    interpretation: str
    limitations: str
    warnings: tuple[str, ...] = ()
    comparison_to_baseline: str = ""
    placebo_estimates: tuple[float, ...] = ()
    placebo_seed: int | None = None
    n_replications: int | None = None
    assignment_rule: str = ""
    tail_fraction: float | None = None
    actual_estimate: float | None = None
    placebo_mean: float | None = None
    placebo_std: float | None = None
    placebo_p05: float | None = None
    placebo_p50: float | None = None
    placebo_p95: float | None = None


@dataclass(frozen=True)
class RobustnessSuite:
    checks: tuple[RobustnessCheck, ...]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for check in self.checks:
            rows.append(
                {
                    "name": check.name,
                    "status": check.status,
                    "estimate": check.estimate,
                    "std_error": check.std_error,
                    "ci_low": check.ci_low,
                    "ci_high": check.ci_high,
                    "p_value": check.p_value,
                    "n_obs": check.n_obs,
                    "n_regions": check.n_regions,
                    "n_treated_regions": check.n_treated_regions,
                    "month_min": check.month_min,
                    "month_max": check.month_max,
                    "failure_reason": check.failure_reason,
                    "placebo_mean": check.placebo_mean,
                    "tail_fraction": check.tail_fraction,
                }
            )
        return pd.DataFrame(rows)


def run_robustness(
    df: pd.DataFrame,
    *,
    placebo_reps: int = DEFAULT_PLACEBO_REPS,
    placebo_seed: int = DEFAULT_PLACEBO_SEED,
    window_offsets: tuple[tuple[int, int], ...] = DEFAULT_WINDOW_OFFSETS,
) -> RobustnessSuite:
    """运行基准模型和各项稳健性检验。不会修改 df。"""

    if placebo_reps < 1:
        raise ValueError("虚假处理组的重复次数至少为 1。")
    baseline_fit = fit_did(df)
    baseline = _from_did(
        baseline_fit,
        name="baseline_twfe",
        description="基准模型：地区固定效应、月份固定效应，不加入协变量。",
        interpretation="这是后续设定的对照，不是由 p 值单独决定的最终结论。",
        limitations="常规聚类标准误在有限聚类下可能偏低。",
        window_note=_window_note(baseline_fit.month_min, baseline_fit.month_max, baseline_fit.policy_start_month),
    )
    checks = [
        baseline,
        covariate_check(df, baseline),
        *[
            window_check(df, baseline, pre_months=pre, post_months=post)
            for pre, post in window_offsets
        ],
        placebo_time_check(df, baseline),
        placebo_assignment_check(df, baseline, reps=placebo_reps, seed=placebo_seed),
    ]
    return RobustnessSuite(tuple(checks))


def covariate_check(df: pd.DataFrame, baseline: RobustnessCheck | None = None) -> RobustnessCheck:
    """在双向固定效应模型中加入不受策略影响的 promo_intensity。"""

    name = "covariate_promo_intensity"
    description = "基准双向固定效应模型加入 promo_intensity。不加入 region_baseline，也不加入受策略影响的变量。"
    limitations = (
        "promo_intensity 在数据生成时不依赖处理变量。加入它只能吸收与处理正交的剩余变异，"
        "不能证明没有遗漏变量。region_baseline 是时间不变的地区水平，地区固定效应已经吸收它，"
        "它不提供独立的识别信息。"
    )
    if "promo_intensity" not in df.columns:
        return _failed(name, description, "数据中没有 promo_intensity。", limitations)
    promo = pd.to_numeric(df["promo_intensity"], errors="coerce")
    if promo.isna().any() or not np.isfinite(promo.to_numpy(dtype=float)).all():
        return _failed(name, description, "promo_intensity 含有缺失值或非数值，估计不会删除这些行。", limitations)
    baseline_note = _baseline_absorption_note(df)
    residual = twoway_demean(df.assign(promo_intensity=promo), ["promo_intensity"])
    residual_std = float(residual["promo_intensity"].std(ddof=0))
    if residual_std <= 1e-8:
        return _failed(
            name,
            description,
            f"promo_intensity 在地区和月份固定效应之后的剩余标准差为 {residual_std:.3g}，"
            "没有足够的独立变异，因此没有把它放进回归。",
            limitations + baseline_note,
        )
    try:
        require_panel(df, ("region_id", "month", "outcome", "treated_post"))
        fit = fit_clustered_formula(
            df,
            "outcome ~ treated_post + promo_intensity + C(region_id) + C(month)",
            ("treated_post",),
        )
    except (ValueError, RuntimeError) as exc:
        return _failed(name, description, str(exc), limitations + baseline_note)
    coefficient = fit.coefficient("treated_post")
    n_treated, n_control = region_arm_counts(df)
    check = _success_from_values(
        name=name,
        description=description,
        estimate=coefficient.estimate,
        std_error=coefficient.std_error,
        ci_low=coefficient.ci_low,
        ci_high=coefficient.ci_high,
        p_value=coefficient.p_value,
        n_obs=fit.n_obs,
        n_regions=fit.n_regions,
        n_treated=n_treated,
        n_control=n_control,
        month_min=int(df["month"].min()),
        month_max=int(df["month"].max()),
        policy_start=infer_common_policy_start(df),
        window_note="使用全部可用月份。",
        interpretation=(
            f"promo_intensity 在双向去均值后的剩余标准差为 {residual_std:.4f}，所以它没有被月份固定效应完全吸收。"
            + baseline_note
        ),
        limitations=limitations,
        warnings=fit.warnings,
    )
    return _with_comparison(check, baseline)


def window_check(
    df: pd.DataFrame,
    baseline: RobustnessCheck | None,
    *,
    pre_months: int,
    post_months: int,
) -> RobustnessCheck:
    """估计实施时点两侧指定长度的窗口。窗口必须同时有处理前和处理后。"""

    name = f"window_pre{pre_months}_post{post_months}"
    description = f"只保留真实实施前 {pre_months} 个月和实施后 {post_months} 个月。"
    limitations = "缩短窗口可能减弱更长时期的趋势差异，但月份更少，区间通常更宽，也不能消除窗口内部的偏差。"
    if pre_months < 1 or post_months < 1:
        return _failed(name, description, "替代窗口必须同时包含处理前和处理后时期。", limitations)
    try:
        policy_start = infer_common_policy_start(df)
    except ValueError as exc:
        return _failed(name, description, str(exc), limitations)
    month_min = policy_start - pre_months
    month_max = policy_start + post_months - 1
    available = set(pd.to_numeric(df["month"], errors="coerce").dropna().astype(int))
    requested = set(range(month_min, month_max + 1))
    if not requested.issubset(available):
        return _failed(
            name,
            description,
            f"请求的月份 {month_min}–{month_max} 超出当前样本，没有截断后勉强估计。",
            limitations,
            month_min=month_min,
            month_max=month_max,
            policy_start_month=policy_start,
        )
    sample = df.loc[df["month"].between(month_min, month_max)].copy(deep=True)
    pre = sample.loc[sample["month"] < policy_start]
    post = sample.loc[sample["month"] >= policy_start]
    if pre.empty or post.empty:
        return _failed(
            name,
            description,
            "窗口内缺少处理前或处理后时期，不能当作双重差分。",
            limitations,
            month_min=month_min,
            month_max=month_max,
            policy_start_month=policy_start,
            n_obs=len(sample),
        )
    try:
        fit = fit_did(sample)
    except (ValueError, RuntimeError) as exc:
        return _failed(name, description, str(exc), limitations, month_min=month_min, month_max=month_max)
    check = _from_did(
        fit,
        name=name,
        description=description,
        interpretation=(
            f"该窗口有处理前 {pre['month'].nunique()} 个月、处理后 {post['month'].nunique()} 个月。"
            "样本量下降会降低精度。"
        ),
        limitations=limitations,
        window_note=_window_note(month_min, month_max, policy_start),
    )
    return _with_comparison(check, baseline)


def placebo_time_check(df: pd.DataFrame, baseline: RobustnessCheck | None = None) -> RobustnessCheck:
    """把虚假实施月份放在真实实施之前，并且只使用真实实施前的观测。"""

    name = "placebo_time"
    description = "在真实策略实施之前指定一个虚假实施月份，处理组身份保持不变。"
    limitations = (
        "该检验只描述真实实施前窗口里的虚假政策效应。"
        "它不包含任何真实处理后的观测，因此不能检查实施后的混杂，也不能证明平行趋势。"
    )
    try:
        policy_start = infer_common_policy_start(df)
    except ValueError as exc:
        return _failed(name, description, str(exc), limitations)
    pre_months = sorted(int(value) for value in df.loc[df["month"] < policy_start, "month"].unique())
    if len(pre_months) < 2:
        return _failed(
            name,
            description,
            "真实实施前不足两个月，无法同时形成虚假处理前和虚假处理后。",
            limitations,
            policy_start_month=policy_start,
        )
    fake_start = pre_months[len(pre_months) // 2]
    if fake_start == pre_months[0]:
        return _failed(name, description, "虚假实施月份之前没有时期，不能估计安慰剂双重差分。", limitations)
    sample = df.loc[df["month"] < policy_start].copy(deep=True)
    if sample.empty or bool((sample["month"] >= policy_start).any()):
        raise RuntimeError("安慰剂时间样本错误地包含了真实处理后的观测。")
    sample["post"] = (sample["month"] >= fake_start).astype(int)
    sample["treated_post"] = sample["treated"].astype(int) * sample["post"].astype(int)
    sample["treatment_start_month"] = pd.Series(pd.NA, index=sample.index, dtype="Int64")
    sample.loc[sample["treated"].astype(int) == 1, "treatment_start_month"] = int(fake_start)
    if sample["post"].nunique() < 2:
        return _failed(name, description, "虚假时点没有同时产生处理前和处理后。", limitations)
    try:
        fit = fit_did(sample)
    except (ValueError, RuntimeError) as exc:
        return _failed(name, description, str(exc), limitations, policy_start_month=policy_start)
    check = _from_did(
        fit,
        name=name,
        description=description,
        interpretation=(
            f"虚假实施月份是 {fake_start}，估计只用月份 "
            f"{int(sample['month'].min())}–{int(sample['month'].max())}，真实实施月份是 {policy_start}。"
        ),
        limitations=limitations,
        window_note=(
            f"虚假实施月份 {fake_start}；样本月份 "
            f"{int(sample['month'].min())}–{int(sample['month'].max())}，不含真实实施月及之后。"
        ),
        placebo_policy_start=int(fake_start),
        policy_start_override=policy_start,
    )
    return _with_comparison(check, baseline)


def placebo_assignment_check(
    df: pd.DataFrame,
    baseline: RobustnessCheck | None,
    *,
    reps: int,
    seed: int,
) -> RobustnessCheck:
    """按地区类型保持处理名额，重复抽取虚假处理组。"""

    name = "placebo_assignment"
    description = "保持各组处理名额和真实实施月份，随机更换哪些地区被标记为处理。"
    limitations = (
        "随机化虚假处理组给出的是重新分配标签后的估计分布。"
        "尾部比例只描述真实估计在这个分布中的位置，不是平行趋势检验的 p 值，"
        "也不等于一次真实随机实验。重复次数有限，尾部比例的步长是 1/重复次数。"
    )
    try:
        policy_start = infer_common_policy_start(df)
        actual = baseline.estimate if baseline is not None and baseline.estimate is not None else fit_did(df).estimate
    except (ValueError, RuntimeError) as exc:
        return _failed(name, description, str(exc), limitations)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    n_treated = n_control = None
    for _ in range(reps):
        labels = draw_placebo_labels(df, rng)
        sample = _apply_labels(df, labels, policy_start)
        n_treated, n_control = region_arm_counts(sample)
        try:
            estimates.append(fit_did(sample).estimate)
        except (ValueError, RuntimeError) as exc:
            return _failed(name, description, f"虚假处理组估计失败：{exc}", limitations)
    values = np.asarray(estimates, dtype=float)
    tail = float(np.mean(np.abs(values) >= abs(float(actual))))
    check = RobustnessCheck(
        name=name,
        description=description,
        status="success",
        failure_reason=None,
        estimate=None,
        std_error=None,
        ci_low=None,
        ci_high=None,
        p_value=None,
        p_value_reason="不把虚假分配分布的尾部比例报告为识别假设的 p 值。",
        n_obs=len(df),
        n_regions=int(df["region_id"].nunique()),
        n_treated_regions=n_treated,
        n_control_regions=n_control,
        month_min=int(df["month"].min()),
        month_max=int(df["month"].max()),
        policy_start_month=policy_start,
        placebo_policy_start=None,
        window_note=_window_note(int(df["month"].min()), int(df["month"].max()), policy_start),
        interpretation=(
            f"真实估计为 {float(actual):.4f}。{reps} 次虚假分配的均值为 {float(values.mean()):.4f}，"
            f"标准差为 {float(values.std(ddof=1)):.4f}。"
            f"绝对值和真实估计至少一样大的比例为 {tail:.3f}。"
        ),
        limitations=limitations,
        warnings=finite_cluster_warning(int(df["region_id"].nunique()))
        + (("重复次数少于 20，经验分布比较粗。",) if reps < 20 else ()),
        comparison_to_baseline=(
            "这里比较的是真实估计在虚假分配分布中的位置。是否显著不能单独作为稳健性的结论，"
            "这个尾部比例也不能证明识别假设成立。"
        ),
        placebo_estimates=tuple(float(value) for value in values),
        placebo_seed=int(seed),
        n_replications=int(reps),
        assignment_rule=ASSIGNMENT_RULE,
        tail_fraction=tail,
        actual_estimate=float(actual),
        placebo_mean=float(values.mean()),
        placebo_std=float(values.std(ddof=1)) if reps > 1 else 0.0,
        placebo_p05=float(np.quantile(values, 0.05)),
        placebo_p50=float(np.quantile(values, 0.50)),
        placebo_p95=float(np.quantile(values, 0.95)),
    )
    return check


def draw_placebo_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """按地区类型保持现有处理名额，返回地区层面的虚假处理标签。"""

    regions = df.groupby("region_id", sort=False).agg(
        region_type=("region_type", "first"),
        treated=("treated", "max"),
    )
    labels: dict[object, int] = {}
    for region_type in sorted(regions["region_type"].astype(str).unique()):
        group = regions.loc[regions["region_type"].astype(str) == region_type].sort_index()
        region_ids = group.index.to_numpy()
        n_treated = int(pd.to_numeric(group["treated"], errors="coerce").sum())
        if not 0 <= n_treated <= len(region_ids):
            raise ValueError(f"{region_type} 的处理地区数无法用于虚假分配。")
        chosen = set(rng.choice(region_ids, size=n_treated, replace=False).tolist())
        for region_id in region_ids:
            labels[region_id] = int(region_id in chosen)
    return pd.Series(labels, name="treated")


def format_robustness(suite: RobustnessSuite) -> str:
    lines = ["稳健性检验（模拟数据）", "是否显著不能单独作为稳健性的结论。"]
    for check in suite.checks:
        if check.status != "success":
            lines.append(f"{check.name}: 失败。{check.failure_reason}")
            continue
        if check.estimate is None:
            lines.append(
                f"{check.name}: 重复 {check.n_replications} 次，种子 {check.placebo_seed}，"
                f"均值 {check.placebo_mean:.4f}，中位数 {check.placebo_p50:.4f}，"
                f"5%–95% 为 [{check.placebo_p05:.4f}, {check.placebo_p95:.4f}]，"
                f"真实估计 {check.actual_estimate:.4f}，尾部比例 {check.tail_fraction:.3f}。"
            )
        else:
            lines.append(
                f"{check.name}: 估计 {check.estimate:.4f}，95% 区间 "
                f"[{check.ci_low:.4f}, {check.ci_high:.4f}]，"
                f"样本 {check.n_obs} 行 / {check.n_regions} 个地区，"
                f"月份 {check.month_min}–{check.month_max}。"
            )
        if check.comparison_to_baseline:
            lines.append(check.comparison_to_baseline)
    return "\n".join(lines)


def _apply_labels(df: pd.DataFrame, labels: pd.Series, policy_start: int) -> pd.DataFrame:
    out = df.copy(deep=True)
    mapped = out["region_id"].map(labels)
    if mapped.isna().any():
        raise ValueError("虚假处理标签没有覆盖全部地区。")
    out["treated"] = mapped.astype(int)
    out["treated_post"] = out["treated"] * out["post"].astype(int)
    out["treatment_start_month"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[out["treated"] == 1, "treatment_start_month"] = int(policy_start)
    return out


def _baseline_absorption_note(df: pd.DataFrame) -> str:
    if "region_baseline" not in df.columns:
        return " 数据中没有 region_baseline。"
    values = pd.to_numeric(df["region_baseline"], errors="coerce")
    if values.isna().any():
        return " region_baseline 含有缺失或非数值，没有把它当作控制变量。"
    within = values - values.groupby(df["region_id"]).transform("mean")
    within_std = float(within.std(ddof=0))
    return (
        f" region_baseline 的组内标准差为 {within_std:.3g}，"
        "被地区固定效应吸收，没有作为独立控制变量。"
    )


def _from_did(
    result: DIDResult,
    *,
    name: str,
    description: str,
    interpretation: str,
    limitations: str,
    window_note: str,
    placebo_policy_start: int | None = None,
    policy_start_override: int | None = None,
) -> RobustnessCheck:
    return RobustnessCheck(
        name=name,
        description=description,
        status="success",
        failure_reason=None,
        estimate=result.estimate,
        std_error=result.std_error,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        p_value=result.p_value,
        p_value_reason=None,
        n_obs=result.n_obs,
        n_regions=result.n_regions,
        n_treated_regions=result.n_treated_regions,
        n_control_regions=result.n_control_regions,
        month_min=result.month_min,
        month_max=result.month_max,
        policy_start_month=result.policy_start_month if policy_start_override is None else policy_start_override,
        placebo_policy_start=placebo_policy_start,
        window_note=window_note,
        interpretation=interpretation,
        limitations=limitations,
        warnings=result.warnings,
    )


def _success_from_values(
    *,
    name: str,
    description: str,
    estimate: float,
    std_error: float,
    ci_low: float,
    ci_high: float,
    p_value: float,
    n_obs: int,
    n_regions: int,
    n_treated: int,
    n_control: int,
    month_min: int,
    month_max: int,
    policy_start: int,
    window_note: str,
    interpretation: str,
    limitations: str,
    warnings: tuple[str, ...],
) -> RobustnessCheck:
    return RobustnessCheck(
        name=name,
        description=description,
        status="success",
        failure_reason=None,
        estimate=estimate,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        p_value_reason=None,
        n_obs=n_obs,
        n_regions=n_regions,
        n_treated_regions=n_treated,
        n_control_regions=n_control,
        month_min=month_min,
        month_max=month_max,
        policy_start_month=policy_start,
        placebo_policy_start=None,
        window_note=window_note,
        interpretation=interpretation,
        limitations=limitations,
        warnings=warnings,
    )


def _failed(
    name: str,
    description: str,
    reason: str,
    limitations: str,
    *,
    month_min: int | None = None,
    month_max: int | None = None,
    policy_start_month: int | None = None,
    n_obs: int | None = None,
) -> RobustnessCheck:
    return RobustnessCheck(
        name=name,
        description=description,
        status="failed",
        failure_reason=reason,
        estimate=None,
        std_error=None,
        ci_low=None,
        ci_high=None,
        p_value=None,
        p_value_reason=reason,
        n_obs=n_obs,
        n_regions=None,
        n_treated_regions=None,
        n_control_regions=None,
        month_min=month_min,
        month_max=month_max,
        policy_start_month=policy_start_month,
        placebo_policy_start=None,
        window_note="",
        interpretation="该设定没有给出估计值。",
        limitations=limitations,
    )


def _with_comparison(check: RobustnessCheck, baseline: RobustnessCheck | None) -> RobustnessCheck:
    if baseline is None:
        return check
    text = _comparison_text(baseline, check)
    return RobustnessCheck(**{**check.__dict__, "comparison_to_baseline": text})


def _comparison_text(baseline: RobustnessCheck, other: RobustnessCheck) -> str:
    if other.status != "success" or baseline.status != "success" or other.estimate is None or baseline.estimate is None:
        return "该设定没有可与基准系数比较的估计值。是否显著不能单独作为稳健性的结论。"
    difference = other.estimate - baseline.estimate
    same_sign = np.sign(other.estimate) == np.sign(baseline.estimate) or other.estimate == 0 or baseline.estimate == 0
    intervals_overlap = not (other.ci_high < baseline.ci_low or other.ci_low > baseline.ci_high)
    direction = "方向与基准相同。" if same_sign else "方向与基准不同。"
    overlap = "两个 95% 区间有重叠。" if intervals_overlap else "两个 95% 区间没有重叠。"
    return (
        f"估计值与基准相差 {difference:.4f}。{direction}{overlap}"
        f"样本量从 {baseline.n_obs} 行变为 {other.n_obs} 行。"
        "是否显著不能单独作为稳健性的结论。"
    )


def _window_note(month_min: int, month_max: int, policy_start: int | None) -> str:
    policy = "未知" if policy_start is None else str(policy_start)
    return f"月份 {month_min}–{month_max}，统一实施月份 {policy}。"
