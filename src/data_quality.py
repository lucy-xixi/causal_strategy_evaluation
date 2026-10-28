"""地区—月份面板的结构化质量检查。

检查只读取数据。故意构造的异常样本由 copy_with_issue 返回新表，原始表保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import MONTH_MAX, MONTH_MIN, N_REGIONS, POLICY_START_MONTH

REQUIRED_COLUMNS = (
    "region_id",
    "region_type",
    "month",
    "year_month",
    "treated",
    "post",
    "treated_post",
    "region_baseline",
    "promo_intensity",
    "outcome",
)
# 对照地区的 treatment_start_month 允许为空，不放进通用缺失检查。
START_COLUMN = "treatment_start_month"
OUTLIER_IQR_MULTIPLIER = 3.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    severity: str
    message: str
    n_issues: int
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityReport:
    checks: tuple[CheckResult, ...]

    def check(self, name: str) -> CheckResult:
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(name)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.checks if item.severity == "error")

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": item.name,
                    "passed": item.passed,
                    "severity": item.severity,
                    "message": item.message,
                    "n_issues": item.n_issues,
                    "details": item.details,
                }
                for item in self.checks
            ],
        }


def assess_panel(
    df: pd.DataFrame,
    *,
    month_min: int = MONTH_MIN,
    month_max: int = MONTH_MAX,
    policy_start_month: int = POLICY_START_MONTH,
    expected_n_regions: int | None = N_REGIONS,
) -> QualityReport:
    """返回各项检查的结果，不修改 df。"""

    checks = (
        _missing_values(df),
        _duplicate_region_month(df),
        _time_range(df, month_min, month_max),
        _panel_completeness(df, month_min, month_max, expected_n_regions),
        _group_distribution(df),
        _treatment_logic(df, policy_start_month),
        _outlier_diagnostic(df),
    )
    return QualityReport(checks)


def copy_with_issue(df: pd.DataFrame, issue: str) -> pd.DataFrame:
    """返回带有一个故意错误的副本。不修改调用方传入的表。"""

    out = df.copy(deep=True)
    if issue == "duplicate_region_month":
        return pd.concat([out, out.iloc[[0]]], ignore_index=True)
    if issue == "missing_outcome":
        out.loc[out.index[0], "outcome"] = np.nan
        return out
    if issue == "illegal_treated_code":
        out.loc[out.index[0], "treated"] = 2
        return out
    if issue == "illegal_interaction":
        control_index = out.index[out["treated"] == 0][0]
        out.loc[control_index, "treated_post"] = 1
        return out
    if issue == "illegal_post_timing":
        pre_index = out.index[out["month"] < POLICY_START_MONTH][0]
        out.loc[pre_index, "post"] = 1
        out.loc[pre_index, "treated_post"] = int(out.loc[pre_index, "treated"]) * 1
        return out
    if issue == "month_out_of_range":
        out.loc[out.index[0], "month"] = -1
        return out
    if issue == "missing_region_month":
        return out.drop(index=out.index[0]).reset_index(drop=True)
    if issue == "treated_start_misaligned":
        treated_region = out.loc[out["treated"] == 1, "region_id"].iloc[0]
        out.loc[out["region_id"] == treated_region, START_COLUMN] = 99
        return out
    raise ValueError(f"未知的异常类型：{issue}")


def _missing_values(df: pd.DataFrame) -> CheckResult:
    absent = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    na_counts = {
        column: int(df[column].isna().sum())
        for column in REQUIRED_COLUMNS
        if column in df.columns
    }
    n_issues = sum(na_counts.values()) + len(absent)
    return CheckResult(
        name="missing_values",
        passed=n_issues == 0,
        severity="error",
        message="必需列缺失或含有缺失值。" if n_issues else "必需列没有缺失值。",
        n_issues=n_issues,
        details={"absent_columns": absent, "na_counts": na_counts},
    )


def _duplicate_region_month(df: pd.DataFrame) -> CheckResult:
    if not {"region_id", "month"}.issubset(df.columns):
        return CheckResult(
            name="duplicate_region_month",
            passed=False,
            severity="error",
            message="无法检查重复记录，因为缺少地区或月份。",
            n_issues=1,
            details={},
        )
    duplicated = df.duplicated(["region_id", "month"], keep=False)
    n_issues = int(duplicated.sum())
    return CheckResult(
        name="duplicate_region_month",
        passed=n_issues == 0,
        severity="error",
        message="存在重复的地区—月份记录。" if n_issues else "地区—月份没有重复。",
        n_issues=n_issues,
        details={},
    )


def _time_range(df: pd.DataFrame, month_min: int, month_max: int) -> CheckResult:
    if "month" not in df.columns:
        return CheckResult(
            name="time_range",
            passed=False,
            severity="error",
            message="缺少月份列。",
            n_issues=1,
            details={},
        )
    month = pd.to_numeric(df["month"], errors="coerce")
    invalid = month.isna() | (month < month_min) | (month > month_max) | (np.floor(month) != month)
    bad_values = sorted(set(month.loc[invalid].dropna().astype(int).tolist()))
    n_issues = int(invalid.sum())
    return CheckResult(
        name="time_range",
        passed=n_issues == 0,
        severity="error",
        message="存在超出样本窗口或无法解析的月份。" if n_issues else "月份都落在样本窗口内。",
        n_issues=n_issues,
        details={"month_min": month_min, "month_max": month_max, "bad_values": bad_values},
    )


def _panel_completeness(
    df: pd.DataFrame,
    month_min: int,
    month_max: int,
    expected_n_regions: int | None,
) -> CheckResult:
    if not {"region_id", "month"}.issubset(df.columns):
        return CheckResult(
            name="panel_completeness",
            passed=False,
            severity="error",
            message="无法检查面板完整性。",
            n_issues=1,
            details={},
        )
    regions = pd.unique(df["region_id"])
    expected_months = list(range(month_min, month_max + 1))
    full = pd.MultiIndex.from_product([regions, expected_months], names=["region_id", "month"])
    actual = pd.MultiIndex.from_frame(df[["region_id", "month"]].copy())
    missing_pairs = full.difference(actual)
    region_gap = 0 if expected_n_regions is None else abs(int(expected_n_regions) - len(pd.unique(regions)))
    n_issues = int(len(missing_pairs) + region_gap)
    examples = [list(pair) for pair in list(missing_pairs[:5])]
    return CheckResult(
        name="panel_completeness",
        passed=n_issues == 0,
        severity="error",
        message="地区与月份的组合不完整。" if n_issues else "每个地区都有完整的月份组合。",
        n_issues=n_issues,
        details={
            "n_regions": int(len(pd.unique(regions))),
            "expected_n_regions": expected_n_regions,
            "n_missing_pairs": int(len(missing_pairs)),
            "missing_examples": examples,
        },
    )


def _group_distribution(df: pd.DataFrame) -> CheckResult:
    if not {"region_id", "treated", "region_type"}.issubset(df.columns):
        return CheckResult(
            name="group_distribution",
            passed=False,
            severity="error",
            message="无法检查处理组和对照组。",
            n_issues=1,
            details={},
        )
    region_level = df.groupby("region_id", sort=False).agg(
        treated=("treated", "min"),
        treated_max=("treated", "max"),
        region_type=("region_type", "first"),
    )
    consistent = region_level["treated"].eq(region_level["treated_max"])
    n_treated = int((consistent & region_level["treated"].eq(1)).sum())
    n_control = int((consistent & region_level["treated"].eq(0)).sum())
    by_type = (
        region_level.loc[consistent]
        .groupby("region_type")["treated"]
        .agg(n_regions="size", n_treated="sum")
        .astype(int)
        .to_dict(orient="index")
    )
    passed = n_treated > 0 and n_control > 0
    return CheckResult(
        name="group_distribution",
        passed=passed,
        severity="error",
        message="处理组和对照组都存在。" if passed else "处理组或对照组缺失。",
        n_issues=0 if passed else 1,
        details={
            "n_treated_regions": n_treated,
            "n_control_regions": n_control,
            "by_type": by_type,
        },
    )


def _treatment_logic(df: pd.DataFrame, policy_start_month: int) -> CheckResult:
    rules: list[str] = []
    row_flags = pd.Series(False, index=df.index)
    if "treated" not in df.columns or "post" not in df.columns or "treated_post" not in df.columns:
        return CheckResult(
            name="treatment_logic",
            passed=False,
            severity="error",
            message="缺少处理变量，无法检查处理逻辑。",
            n_issues=1,
            details={"rules": ["missing_treatment_columns"]},
        )

    treated = pd.to_numeric(df["treated"], errors="coerce")
    post = pd.to_numeric(df["post"], errors="coerce")
    interaction = pd.to_numeric(df["treated_post"], errors="coerce")
    bad_treated = treated.isna() | ~treated.isin([0, 1])
    bad_post = post.isna() | ~post.isin([0, 1])
    if bad_treated.any():
        rules.append("treated_code")
        row_flags = row_flags | bad_treated.fillna(True)
    if bad_post.any():
        rules.append("post_code")
        row_flags = row_flags | bad_post.fillna(True)

    comparable = treated.notna() & post.notna() & interaction.notna()
    expected_interaction = treated * post
    bad_interaction = comparable & ~np.isclose(
        interaction.to_numpy(dtype=float),
        expected_interaction.to_numpy(dtype=float),
        equal_nan=False,
    )
    if bad_interaction.any():
        rules.append("interaction")
        row_flags = row_flags | bad_interaction.fillna(False)

    if "region_id" in df.columns and df.groupby("region_id")["treated"].nunique(dropna=False).gt(1).any():
        rules.append("treated_varies")
        varying_regions = df.groupby("region_id")["treated"].nunique(dropna=False)
        varying_regions = varying_regions[varying_regions > 1].index
        row_flags = row_flags | df["region_id"].isin(varying_regions)

    if "month" in df.columns:
        month = pd.to_numeric(df["month"], errors="coerce")
        can_time = month.notna() & post.isin([0, 1])
        expected_post = month >= policy_start_month
        bad_timing = can_time & expected_post.ne(post.eq(1))
        if bad_timing.fillna(False).any():
            rules.append("post_timing")
            row_flags = row_flags | bad_timing.fillna(False)

    if START_COLUMN not in df.columns:
        rules.append("missing_start_column")
    else:
        start = pd.to_numeric(df[START_COLUMN], errors="coerce")
        treated_is_one = treated.eq(1)
        treated_is_zero = treated.eq(0)
        bad_treated_start = treated_is_one & (start.isna() | start.ne(policy_start_month))
        bad_control_start = treated_is_zero & start.notna()
        if bad_treated_start.any():
            rules.append("treated_start")
            row_flags = row_flags | bad_treated_start.fillna(False)
        if bad_control_start.any():
            rules.append("control_start")
            row_flags = row_flags | bad_control_start.fillna(False)

    n_issues = int(row_flags.fillna(False).sum())
    return CheckResult(
        name="treatment_logic",
        passed=not rules,
        severity="error",
        message="处理变量与实施时间不一致。" if rules else "处理变量与统一实施月份一致。",
        n_issues=n_issues,
        details={"rules": rules, "policy_start_month": policy_start_month},
    )


def _outlier_diagnostic(df: pd.DataFrame) -> CheckResult:
    if "outcome" not in df.columns:
        return CheckResult(
            name="outlier_diagnostic",
            passed=False,
            severity="info",
            message="没有结果列，不能做异常值诊断。",
            n_issues=1,
            details={"rule": f"Tukey {OUTLIER_IQR_MULTIPLIER} * IQR"},
        )
    outcome = pd.to_numeric(df["outcome"], errors="coerce")
    valid = outcome.dropna()
    if len(valid) < 4:
        return CheckResult(
            name="outlier_diagnostic",
            passed=True,
            severity="info",
            message="有效结果太少，未标记异常值。",
            n_issues=0,
            details={"rule": f"Tukey {OUTLIER_IQR_MULTIPLIER} * IQR"},
        )
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        flagged = outcome.notna() & outcome.ne(q1)
    else:
        low = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        high = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        flagged = outcome.lt(low) | outcome.gt(high)
    n_issues = int(flagged.fillna(False).sum())
    return CheckResult(
        name="outlier_diagnostic",
        passed=n_issues == 0,
        severity="info",
        message="异常值诊断只作提示，不会删除或替换观测。"
        + (f" 当前标记 {n_issues} 行。" if n_issues else " 当前没有标记。"),
        n_issues=n_issues,
        details={
            "rule": f"Tukey {OUTLIER_IQR_MULTIPLIER} * IQR",
            "q1": q1,
            "q3": q3,
        },
    )
