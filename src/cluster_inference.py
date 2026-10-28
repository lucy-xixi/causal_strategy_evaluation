"""地区聚类 OLS 的共用推断工具。

基础双重差分仍由 src.did_analysis.fit_did 估计。这里给事件研究、
协变量设定和组间比较提供同样的聚类协方差与 t(G-1) 区间，避免各写一套口径。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.config import CI_ALPHA, CLUSTER_CAUTION_BELOW


@dataclass(frozen=True)
class Coefficient:
    term: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class HypothesisTest:
    name: str
    null_hypothesis: str
    status: str
    method: str
    statistic: float | None
    df_numerator: int | None
    df_denominator: int | None
    p_value: float | None
    terms: tuple[str, ...]
    reason: str | None
    interpretation: str


@dataclass(frozen=True)
class ClusterFit:
    formula: str
    coefficients: tuple[Coefficient, ...]
    n_obs: int
    n_regions: int
    t_df: int
    t_critical: float
    ci_level: float
    covariance_type: str
    exog_names: tuple[str, ...]
    warnings: tuple[str, ...]
    fitted: object

    def coefficient(self, term: str) -> Coefficient:
        for item in self.coefficients:
            if item.term == term:
                return item
        raise KeyError(term)


def require_panel(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """检查估计所需的输入。不删除、不填补、不改写调用方的数据。"""

    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是 pandas DataFrame。")
    if df.empty:
        raise ValueError("样本为空，无法估计。")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"缺少估计所需的列：{missing}")
    if df.duplicated(["region_id", "month"]).any():
        raise ValueError("存在重复的地区—月份。估计不会从重复记录中挑选一行。")
    numeric_columns = [column for column in columns if column not in {"region_type", "year_month"}]
    if df[numeric_columns].isna().any().any():
        raise ValueError("估计列中有缺失值。估计不会删除这些观测。")


def region_arm_counts(df: pd.DataFrame) -> tuple[int, int]:
    summary = df.groupby("region_id", sort=False)["treated"].agg(["min", "max"])
    consistent = summary["min"].eq(summary["max"])
    n_treated = int((consistent & summary["min"].eq(1)).sum())
    n_control = int((consistent & summary["min"].eq(0)).sum())
    return n_treated, n_control


def infer_common_policy_start(df: pd.DataFrame) -> int:
    """读取统一实施月份。实施月份不一致时直接拒绝，不把它当成交错采纳模型。"""

    if "treatment_start_month" not in df.columns or "treated" not in df.columns:
        raise ValueError("缺少 treated 或 treatment_start_month，无法确认处理时点。")
    treated = df.loc[pd.to_numeric(df["treated"], errors="coerce").eq(1), ["region_id", "treatment_start_month"]]
    if treated.empty:
        raise ValueError("没有处理地区，无法确认处理时点。")
    if treated["treatment_start_month"].isna().any():
        raise ValueError("处理地区缺少实施月份。")
    by_region = treated.drop_duplicates()
    if by_region["region_id"].duplicated().any():
        raise ValueError("同一处理地区有多个实施月份。当前估计只支持统一处理时点。")
    starts = by_region["treatment_start_month"].unique()
    if len(starts) != 1:
        raise ValueError(
            "处理地区的实施月份不一致。当前事件研究和安慰剂只支持统一处理时点，"
            "不能解释为交错采纳下的动态效应估计。"
        )
    return int(starts[0])


def finite_cluster_warning(n_regions: int) -> tuple[str, ...]:
    notes = [
        (
            f"常规聚类稳健标准误使用的聚类数为 {n_regions}，t 自由度为 {n_regions - 1}。"
            "有限聚类下这个标准误可能偏低，这里没有做 CR2 或 wild cluster bootstrap，"
            "不能视为已经解决了推断问题。"
        )
    ]
    if n_regions < CLUSTER_CAUTION_BELOW:
        notes.append("聚类数少于 30，常规聚类区间的覆盖概率可能明显偏离名义水平。")
    return tuple(notes)


def twoway_demean(df: pd.DataFrame, columns: list[str], *, max_iter: int = 40) -> pd.DataFrame:
    """对列做地区和时间双向去均值，用于检查固定效应之后是否还有剩余变异。"""

    out = df.loc[:, columns].apply(pd.to_numeric, errors="coerce").astype(float)
    if out.isna().any().any():
        raise ValueError("双向去均值的列含有非数值或缺失值。")
    region = df["region_id"]
    month = df["month"]
    previous = out.to_numpy()
    for _ in range(max_iter):
        out = out.sub(out.groupby(region).transform("mean"))
        out = out.sub(out.groupby(month).transform("mean"))
        current = out.to_numpy()
        if np.nanmax(np.abs(current - previous)) < 1e-10:
            break
        previous = current
    return out


def fit_clustered_formula(
    df: pd.DataFrame,
    formula: str,
    focus_terms: tuple[str, ...],
    *,
    alpha: float = CI_ALPHA,
) -> ClusterFit:
    """拟合公式并只提取关注系数的聚类方差。不丢弃观测。"""

    if not focus_terms:
        raise ValueError("至少需要一个关注系数。")
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须介于 0 和 1 之间。")
    model = smf.ols(formula, data=df, missing="raise")
    exog = np.asarray(model.exog, dtype=float)
    rank = int(np.linalg.matrix_rank(exog))
    if rank != exog.shape[1]:
        raise RuntimeError(
            "设计矩阵不满秩。为避免统计软件自动删除系数，已停止拟合。"
            f"秩为 {rank}，列数为 {exog.shape[1]}。"
        )
    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": df["region_id"].to_numpy()})
    if int(fitted.nobs) != len(df):
        raise RuntimeError("模型使用的样本量与输入不一致。拟合不应丢弃观测。")
    missing_terms = [term for term in focus_terms if term not in fitted.params.index]
    if missing_terms:
        raise RuntimeError(f"模型没有返回这些系数，估计已停止：{missing_terms}")

    n_regions = int(df["region_id"].nunique())
    t_df = n_regions - 1
    if t_df <= 0:
        raise ValueError("至少需要两个地区才能按地区聚类并做 t 推断。")
    t_critical = float(stats.t.ppf(1 - alpha / 2, t_df))
    coefficients = tuple(_coefficient(fitted, term, t_critical, t_df) for term in focus_terms)
    return ClusterFit(
        formula=formula,
        coefficients=coefficients,
        n_obs=int(fitted.nobs),
        n_regions=n_regions,
        t_df=t_df,
        t_critical=t_critical,
        ci_level=1 - alpha,
        covariance_type=str(fitted.cov_type),
        exog_names=tuple(fitted.params.index.astype(str)),
        warnings=finite_cluster_warning(n_regions),
        fitted=fitted,
    )


def wald_test_zero(
    fitted: object,
    terms: tuple[str, ...],
    *,
    n_regions: int,
    name: str,
    null_hypothesis: str,
    interpretation: str,
) -> HypothesisTest:
    """检验一组系数同时为零。协方差子矩阵不可逆时返回失败状态，不填造 p 值。"""

    method = "聚类稳健 Wald 统计量除以约束个数，与 F(q, G-1) 比较；原假设是所列系数同时为零。"
    if len(terms) == 0:
        return _failed_test(name, null_hypothesis, method, interpretation, "没有可纳入联合检验的系数。")
    params = fitted.params
    missing = [term for term in terms if term not in params.index]
    if missing:
        return _failed_test(name, null_hypothesis, method, interpretation, f"这些系数没有被估计：{missing}")
    covariance = fitted.cov_params().loc[list(terms), list(terms)].to_numpy(dtype=float)
    estimate = params.loc[list(terms)].to_numpy(dtype=float)
    if not np.isfinite(covariance).all() or not np.isfinite(estimate).all():
        return _failed_test(name, null_hypothesis, method, interpretation, "系数或协方差含有非有限数值。")
    symmetrized = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetrized)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) <= 1e-8 * scale:
        return _failed_test(
            name,
            null_hypothesis,
            method,
            interpretation,
            "这些系数的聚类协方差子矩阵不可逆，联合检验无法可靠执行。",
        )
    df_denominator = n_regions - 1
    if df_denominator <= 0:
        return _failed_test(name, null_hypothesis, method, interpretation, "聚类数不足，无法计算 F 分布自由度。")
    try:
        solved = np.linalg.solve(symmetrized, estimate)
    except np.linalg.LinAlgError:
        return _failed_test(name, null_hypothesis, method, interpretation, "协方差子矩阵无法求逆。")
    q = len(terms)
    statistic = float(estimate @ solved) / q
    if not np.isfinite(statistic):
        return _failed_test(name, null_hypothesis, method, interpretation, "Wald 统计量不是有限数值。")
    p_value = float(stats.f.sf(statistic, q, df_denominator))
    if not np.isfinite(p_value):
        return _failed_test(name, null_hypothesis, method, interpretation, "p 值不是有限数值。")
    return HypothesisTest(
        name=name,
        null_hypothesis=null_hypothesis,
        status="success",
        method=method,
        statistic=statistic,
        df_numerator=q,
        df_denominator=df_denominator,
        p_value=p_value,
        terms=tuple(terms),
        reason=None,
        interpretation=interpretation,
    )


def _coefficient(fitted: object, term: str, t_critical: float, t_df: int) -> Coefficient:
    estimate = float(fitted.params[term])
    variance = float(fitted.cov_params().loc[term, term])
    if np.isfinite(variance) and variance >= 0:
        std_error = float(np.sqrt(variance))
    elif np.isfinite(variance) and variance > -1e-10:
        std_error = 0.0
    else:
        std_error = float("nan")
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
    return Coefficient(term, estimate, std_error, ci_low, ci_high, p_value)


def _failed_test(
    name: str,
    null_hypothesis: str,
    method: str,
    interpretation: str,
    reason: str,
) -> HypothesisTest:
    return HypothesisTest(
        name=name,
        null_hypothesis=null_hypothesis,
        status="failed",
        method=method,
        statistic=None,
        df_numerator=None,
        df_denominator=None,
        p_value=None,
        terms=(),
        reason=reason,
        interpretation=interpretation,
    )
