"""基础双重差分在本模拟设计下的估计目标。

估计函数本身不从这里读取真值。调用方在知道数据生成机制时，才把类型效应传进来做对照。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def design_target(specs: Iterable[object]) -> float:
    """按处理地区数加权的静态 ATT。

    每个 spec 需要有 n_treated 和 tau。权重是该类型的处理地区数，
    不是类型个数，也不是该类型的全部地区数。
    """

    treated = 0
    total = 0.0
    for spec in specs:
        n_treated = int(spec.n_treated)
        if n_treated < 0:
            raise ValueError("处理地区数不能为负。")
        treated += n_treated
        total += n_treated * float(spec.tau)
    if treated == 0:
        raise ValueError("没有处理地区，无法定义处理组平均效应。")
    return total / treated


def att_target_from_panel(
    df: pd.DataFrame,
    tau_by_type: Mapping[str, float],
) -> float:
    """用样本中实际出现的处理地区计算等权 ATT。

    每个处理地区计一次。调用前应确认同一地区只有一个 region_type。
    只有在平衡面板、统一实施时点、处理效应不随事件时间变化时，
    这个平均数才等于双向固定效应中的 treated_post 系数。
    """

    treated = df.loc[df["treated"] == 1, ["region_id", "region_type"]].drop_duplicates()
    if treated.empty:
        raise ValueError("没有处理地区，无法计算处理组平均效应。")
    if treated["region_id"].duplicated().any():
        raise ValueError("同一地区有多个 region_type，不能计算处理地区等权效应。")
    missing = sorted(set(treated["region_type"]) - set(tau_by_type))
    if missing:
        raise ValueError(f"缺少这些地区类型的处理效应：{missing}")
    effects = treated["region_type"].map(tau_by_type)
    if effects.isna().any():
        raise ValueError("部分处理地区没有对应的处理效应。")
    return float(effects.mean())


def equal_region_weight_issues(df: pd.DataFrame) -> list[str]:
    """列出双向固定效应不再等于处理地区等权 ATT 的设计问题。

    返回空列表表示：平衡面板、处理状态在地区内恒定、有处理和对照、
    有处理前和处理后，且所有处理地区共享同一个实施月份，post 与该月份一致。
    这里不能证明处理效应本身不随时间变化；那是数据生成假设。
    """

    issues: list[str] = []
    required = {"region_id", "month", "treated", "post", "treatment_start_month"}
    missing = sorted(required - set(df.columns))
    if missing:
        return [f"缺少判断估计目标所需的列：{missing}"]
    if df.empty:
        return ["样本为空。"]
    if df.duplicated(["region_id", "month"]).any():
        issues.append("存在重复的地区—月份，处理地区不再等权。")

    treated_values = pd.to_numeric(df["treated"], errors="coerce")
    if treated_values.isna().any() or not treated_values.isin([0, 1]).all():
        issues.append("treated 含有 0/1 以外的取值。")
        return issues

    nunique_treated = df.groupby("region_id", sort=False)["treated"].nunique(dropna=False)
    if (nunique_treated > 1).any():
        issues.append("处理状态在地区内不恒定。")

    n_regions = int(df["region_id"].nunique())
    n_months = int(df["month"].nunique())
    duplicate_keys = df.duplicated(["region_id", "month"]).any()
    if n_regions * n_months != len(df) and not duplicate_keys:
        issues.append("面板不平衡，双向固定效应不再对处理地区等权。")

    if not {0, 1}.issubset(set(treated_values.astype(int))):
        issues.append("样本需要同时包含处理地区和对照地区。")

    treated_rows = df.loc[treated_values == 1, ["region_id", "treatment_start_month"]].drop_duplicates()
    if treated_rows.empty:
        issues.append("没有处理地区。")
    elif treated_rows["region_id"].duplicated().any() or treated_rows["treatment_start_month"].isna().any():
        issues.append("处理地区的实施月份不统一。")
    elif treated_rows["treatment_start_month"].nunique(dropna=False) != 1:
        issues.append("处理地区的实施月份不统一。")
    else:
        start = treated_rows["treatment_start_month"].iloc[0]
        expected_post = (df["month"] >= start).astype(int)
        observed_post = pd.to_numeric(df["post"], errors="coerce")
        if observed_post.isna().any() or not expected_post.eq(observed_post.astype(int)).all():
            issues.append("post 与统一实施月份不一致。")

    post_values = pd.to_numeric(df["post"], errors="coerce")
    if post_values.dropna().nunique() < 2:
        issues.append("样本需要同时包含处理前和处理后时期。")
    return issues
