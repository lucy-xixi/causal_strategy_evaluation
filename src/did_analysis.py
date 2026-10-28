"""基础双重差分：地区固定效应、月份固定效应、按地区聚类的标准误。

估计函数不改正数据。缺失值、重复键和非法处理变量会直接报错，而不是被删掉或重写。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.config import CI_ALPHA, CLUSTER_CAUTION_BELOW
from src.estimand import att_target_from_panel, equal_region_weight_issues

FORMULA = "outcome ~ treated_post + C(region_id) + C(month)"
MODEL_COLUMNS = ("outcome", "treated_post", "region_id", "month")


@dataclass(frozen=True)
class DIDResult:
    """一次基础双重差分的估计与样本范围。"""

    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    ci_level: float
    t_critical: float
    t_df: int
    n_obs: int
    n_dropped: int
    n_regions: int
    n_treated_regions: int
    n_control_regions: int
    n_months: int
    n_pre_months: int
    n_post_months: int
    month_min: int
    month_max: int
    policy_start_month: int | None
    target_effect: float | None
    bias: float | None
    estimand_matches_target: bool
    covariance_type: str
    specification: str
    exog_names: tuple[str, ...]
    warnings: tuple[str, ...]
    inference_note: str


def fit_did(
    df: pd.DataFrame,
    *,
    tau_by_type: Mapping[str, float] | None = None,
    alpha: float = CI_ALPHA,
) -> DIDResult:
    """拟合静态双向固定效应双重差分。

    回归使用调用方给出的 treated_post，不根据业务规则重写处理变量。
    tau_by_type 只用于和预设真值对照；不传入时，结果里没有目标效应和偏差。
    """

    _reject_unusable(df)
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须介于 0 和 1 之间。")

    model = smf.ols(FORMULA, data=df, missing="raise")
    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": df["region_id"].to_numpy()})
    if int(fitted.nobs) != len(df):
        raise RuntimeError("模型使用的样本量与输入不一致。拟合过程不应丢弃观测。")
    if "treated_post" not in fitted.params.index:
        raise RuntimeError("模型没有返回 treated_post 系数。")

    estimate = float(fitted.params["treated_post"])
    # 只取目标系数的方差。虚拟变量很多时，整条协方差对角线可能有数值上的负值，
    # 对全部对角线开方会发出警告，但 treated_post 这一项仍然可用。
    variance = float(fitted.cov_params().loc["treated_post", "treated_post"])
    if np.isfinite(variance) and variance >= 0:
        std_error = float(np.sqrt(variance))
    elif np.isfinite(variance) and variance > -1e-10:
        std_error = 0.0
    else:
        std_error = float("nan")

    n_regions = int(df["region_id"].nunique())
    t_df = n_regions - 1
    if t_df <= 0:
        raise ValueError("至少需要两个地区才能按地区聚类并做 t 推断。")
    t_critical = float(stats.t.ppf(1 - alpha / 2, t_df))
    if np.isfinite(std_error):
        ci_low = estimate - t_critical * std_error
        ci_high = estimate + t_critical * std_error
        if std_error == 0:
            p_value = 1.0 if estimate == 0 else 0.0
        else:
            p_value = float(2 * stats.t.sf(abs(estimate / std_error), t_df))
    else:
        ci_low = float("nan")
        ci_high = float("nan")
        p_value = float("nan")

    issues = equal_region_weight_issues(df)
    target_effect = None if tau_by_type is None else att_target_from_panel(df, tau_by_type)
    matches = target_effect is not None and not issues
    bias = estimate - target_effect if matches else None

    n_treated_regions, n_control_regions = _region_counts(df)
    policy_start = _policy_start(df)
    warnings = _warnings(n_regions, t_df, issues, std_error)
    note = _inference_note(n_regions, t_df)

    return DIDResult(
        estimate=estimate,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        ci_level=1 - alpha,
        t_critical=t_critical,
        t_df=t_df,
        n_obs=int(fitted.nobs),
        n_dropped=0,
        n_regions=n_regions,
        n_treated_regions=n_treated_regions,
        n_control_regions=n_control_regions,
        n_months=int(df["month"].nunique()),
        n_pre_months=int(df.loc[df["post"] == 0, "month"].nunique()),
        n_post_months=int(df.loc[df["post"] == 1, "month"].nunique()),
        month_min=int(df["month"].min()),
        month_max=int(df["month"].max()),
        policy_start_month=policy_start,
        target_effect=target_effect,
        bias=None if bias is None else float(bias),
        estimand_matches_target=matches,
        covariance_type=str(fitted.cov_type),
        specification=FORMULA,
        exog_names=tuple(fitted.params.index.astype(str)),
        warnings=tuple(warnings),
        inference_note=note,
    )


def format_did_summary(result: DIDResult) -> str:
    """把一次估计写成可核对的文本。"""

    lines = [
        "基础双重差分（模拟数据）",
        f"模型：{result.specification}",
        "固定效应：地区固定效应、月份固定效应。主模型不包含协变量。",
        f"系数 treated_post = {result.estimate:.6f}",
        f"聚类标准误 = {result.std_error:.6f}",
        f"{result.ci_level:.0%} 置信区间 = [{result.ci_low:.6f}, {result.ci_high:.6f}]",
        f"p 值 = {result.p_value:.6g}",
        (
            "有效样本："
            f"{result.n_obs} 个地区—月份观测，{result.n_regions} 个地区 "
            f"（处理 {result.n_treated_regions}，对照 {result.n_control_regions}），"
            f"月份 {result.month_min}–{result.month_max}，"
            f"处理前 {result.n_pre_months} 个月，处理后 {result.n_post_months} 个月。"
        ),
        f"拟合时删除的观测数 = {result.n_dropped}",
        result.inference_note,
    ]
    if result.policy_start_month is not None:
        lines.append(f"处理地区的实施月份 = {result.policy_start_month}")
    if result.target_effect is not None:
        lines.append(f"处理地区等权目标效应 = {result.target_effect:.6f}")
        if result.bias is not None:
            lines.append(f"估计值减去目标效应 = {result.bias:.6f}")
        else:
            lines.append("当前样本不满足等权 ATT 的设计条件，因此不报告相对目标效应的偏差。")
    for warning in result.warnings:
        lines.append(f"注意：{warning}")
    return "\n".join(lines)


def _reject_unusable(df: pd.DataFrame) -> None:
    missing = [column for column in ("outcome", "treated", "post", *MODEL_COLUMNS) if column not in df.columns]
    if missing:
        raise ValueError(f"缺少估计所需的列：{missing}")
    if df[list(MODEL_COLUMNS)].isna().any().any():
        raise ValueError("估计列中有缺失值。拟合不会删除这些观测；请先查看数据质量报告。")
    if df.duplicated(["region_id", "month"]).any():
        raise ValueError("存在重复的地区—月份。拟合不会从重复记录中挑选一行。")

    treated = pd.to_numeric(df["treated"], errors="coerce")
    post = pd.to_numeric(df["post"], errors="coerce")
    interaction = pd.to_numeric(df["treated_post"], errors="coerce")
    if treated.isna().any() or not treated.isin([0, 1]).all():
        raise ValueError("treated 只能是 0 或 1。拟合不会改正非法取值。")
    if post.isna().any() or not post.isin([0, 1]).all():
        raise ValueError("post 只能是 0 或 1。拟合不会改正非法取值。")
    if not np.isclose(
        interaction.to_numpy(dtype=float),
        (treated * post).to_numpy(dtype=float),
    ).all():
        raise ValueError("treated_post 与 treated * post 不一致。拟合不会重写处理变量。")
    if treated.nunique() < 2 or post.nunique() < 2:
        raise ValueError("样本必须同时包含处理组、对照组、处理前和处理后。")


def _region_counts(df: pd.DataFrame) -> tuple[int, int]:
    region_treated = df.groupby("region_id", sort=False)["treated"].agg(["min", "max"])
    consistent = region_treated["min"].eq(region_treated["max"])
    n_treated = int((consistent & region_treated["min"].eq(1)).sum())
    n_control = int((consistent & region_treated["min"].eq(0)).sum())
    return n_treated, n_control


def _policy_start(df: pd.DataFrame) -> int | None:
    if "treatment_start_month" not in df.columns:
        return None
    starts = df.loc[df["treated"] == 1, "treatment_start_month"].dropna().unique()
    if len(starts) != 1:
        return None
    return int(starts[0])


def _warnings(n_regions: int, t_df: int, issues: list[str], std_error: float) -> list[str]:
    warnings = [
        (
            f"常规聚类稳健标准误使用的聚类数为 {n_regions}，t 自由度为 {t_df}。"
            "有限聚类下这个标准误可能偏低，本结果没有做 CR2 或 wild cluster bootstrap 校正，"
            "不能视为已经解决了推断问题。"
        )
    ]
    if n_regions < CLUSTER_CAUTION_BELOW:
        warnings.append("聚类数少于 30，常规聚类区间的覆盖概率可能明显偏离名义水平。")
    if std_error == 0:
        warnings.append("残差方差在数值上为零，标准误记为 0，不能据此做推断。")
    elif not np.isfinite(std_error):
        warnings.append("聚类标准误不是有限数值，置信区间和 p 值无法使用。")
    warnings.extend(issues)
    return warnings


def _inference_note(n_regions: int, t_df: int) -> str:
    return (
        f"协方差类型为 cluster，聚类变量是 region_id，G={n_regions}。"
        f"置信区间和双侧 p 值按 t({t_df}) 计算。"
        "地区和月份虚拟变量的个数多于聚类数，全部系数的聚类协方差不满秩；报告的标准误只来自 treated_post 这一项。"
        "在平衡面板和统一实施时点下，该系数对应处理地区的等权平均效应。"
    )
