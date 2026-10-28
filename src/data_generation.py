"""生成可复现的地区—月份面板。

噪声和结构项分开抽样：同一个种子下，structural=True 与普通样本的处理分配相同。
本模块不根据估计结果修改数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BLOCK_LENGTH,
    CALENDAR_ORIGIN,
    COMMON_SHOCK_SD,
    COVARIATE_NOISE_SD,
    DEFAULT_SEED,
    MONTH_MIN,
    N_MONTHS,
    OUTCOME_NOISE_SD,
    POLICY_START_MONTH,
    PROMO_COEF_IN_OUTCOME,
    PROMO_REGION_NOISE_SD,
    PROMO_SEASON_SIN,
    PROMO_TREND_PER_MONTH,
    REGION_BLOCK_SHOCK_SD,
    REGION_INTERCEPT_SD,
    REGION_TYPES,
    SEASON_COS,
    SEASON_SIN,
    TREND_PER_MONTH,
    validate_design,
)

PANEL_COLUMNS = (
    "region_id",
    "region_type",
    "month",
    "year_month",
    "treated",
    "post",
    "treated_post",
    "treatment_start_month",
    "region_baseline",
    "promo_intensity",
    "outcome",
)


def seasonal_component(month: np.ndarray) -> np.ndarray:
    """所有地区共享的季节项。只依赖月份在一年中的位置。"""

    angle = 2.0 * np.pi * (np.asarray(month) % 12) / 12.0
    return SEASON_SIN * np.sin(angle) + SEASON_COS * np.cos(angle)


def common_time_index(month: np.ndarray, common_shock: np.ndarray) -> np.ndarray:
    """趋势、季节性和当月共同冲击。common_shock 的长度必须等于月份数。"""

    month = np.asarray(month)
    shock = np.asarray(common_shock, dtype=float)
    if len(shock) != len(month):
        raise ValueError("共同冲击必须按月份给出，且与 month 一一对应。")
    return TREND_PER_MONTH * month + seasonal_component(month) + shock


def generate_panel(
    seed: int = DEFAULT_SEED,
    *,
    structural: bool = False,
    outcome_noise_sd: float | None = None,
    covariate_noise_sd: float | None = None,
    common_shock_sd: float | None = None,
    region_intercept_sd: float | None = None,
    region_block_shock_sd: float | None = None,
    promo_region_noise_sd: float | None = None,
) -> pd.DataFrame:
    """生成平衡的地区—月份面板。

    structural=True 时关掉地区噪声、共同冲击、协变量噪声和结果噪声，
    保留地区类型基线、共同趋势、季节性、促销的确定性部分，以及处理效应。
    处理分配仍由 seed 决定。
    """

    validate_design()
    assign_rng, noise_rng = _rng_pair(seed)
    regions = _assign_regions(
        assign_rng,
        region_intercept_sd=0.0 if structural else _or_default(region_intercept_sd, REGION_INTERCEPT_SD),
        promo_region_noise_sd=0.0 if structural else _or_default(promo_region_noise_sd, PROMO_REGION_NOISE_SD),
    )
    months = _month_frame(
        noise_rng,
        common_shock_sd=0.0 if structural else _or_default(common_shock_sd, COMMON_SHOCK_SD),
    )
    panel = regions.merge(months, how="cross")
    panel = panel.sort_values(["region_id", "month"]).reset_index(drop=True)
    panel["treated_post"] = (panel["treated"] * panel["post"]).astype(int)

    n_regions = len(regions)
    covariate_sd = 0.0 if structural else _or_default(covariate_noise_sd, COVARIATE_NOISE_SD)
    outcome_sd = 0.0 if structural else _or_default(outcome_noise_sd, OUTCOME_NOISE_SD)
    block_sd = 0.0 if structural else _or_default(region_block_shock_sd, REGION_BLOCK_SHOCK_SD)

    promo_noise = noise_rng.normal(0.0, covariate_sd, size=(n_regions, N_MONTHS)).ravel()
    panel["promo_intensity"] = (
        panel["promo_level"]
        + panel["promo_region_noise"]
        + PROMO_TREND_PER_MONTH * panel["month"]
        + PROMO_SEASON_SIN * np.sin(2.0 * np.pi * (panel["month"] % 12) / 12.0)
        + promo_noise
    )

    n_blocks = N_MONTHS // BLOCK_LENGTH
    block_draws = noise_rng.normal(0.0, block_sd, size=(n_regions, n_blocks))
    block_index = panel["month"].to_numpy() // BLOCK_LENGTH
    region_index = panel["region_id"].to_numpy()
    block_shock = block_draws[region_index, block_index]
    outcome_noise = noise_rng.normal(0.0, outcome_sd, size=(n_regions, N_MONTHS)).ravel()

    panel["outcome"] = (
        panel["region_baseline"]
        + panel["common_time"]
        + PROMO_COEF_IN_OUTCOME * panel["promo_intensity"]
        + panel["tau"] * panel["treated_post"]
        + block_shock
        + outcome_noise
    )
    # tau 是生成机制的内部参数，不放进分析用的列，避免被误当作回归变量。
    result = panel.loc[:, list(PANEL_COLUMNS)].copy()
    if len(result) != len(regions) * N_MONTHS:
        raise RuntimeError("生成的面板不是完整的地区—月份网格。")
    return result


def _rng_pair(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    assignment, noise = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(assignment), np.random.default_rng(noise)


def _or_default(value: float | None, default: float) -> float:
    return default if value is None else float(value)


def _assign_regions(
    rng: np.random.Generator,
    *,
    region_intercept_sd: float,
    promo_region_noise_sd: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    region_id = 0
    for spec in REGION_TYPES:
        for _ in range(spec.n_regions):
            rows.append(
                {
                    "region_id": region_id,
                    "region_type": spec.name,
                    "type_baseline": spec.baseline_outcome,
                    "promo_level": spec.promo_level,
                    "tau": spec.tau,
                }
            )
            region_id += 1
    regions = pd.DataFrame(rows)
    regions["region_baseline"] = regions["type_baseline"] + rng.normal(
        0.0, region_intercept_sd, size=len(regions)
    )
    regions["promo_region_noise"] = rng.normal(0.0, promo_region_noise_sd, size=len(regions))

    treated = np.zeros(len(regions), dtype=int)
    for spec in REGION_TYPES:
        candidates = regions.index[regions["region_type"] == spec.name].to_numpy()
        chosen = rng.choice(candidates, size=spec.n_treated, replace=False)
        treated[chosen] = 1
    regions["treated"] = treated
    regions["treatment_start_month"] = pd.Series(pd.NA, index=regions.index, dtype="Int64")
    regions.loc[regions["treated"] == 1, "treatment_start_month"] = POLICY_START_MONTH
    return regions


def _month_frame(rng: np.random.Generator, *, common_shock_sd: float) -> pd.DataFrame:
    month = np.arange(MONTH_MIN, MONTH_MIN + N_MONTHS)
    shock = rng.normal(0.0, common_shock_sd, size=N_MONTHS)
    origin = pd.Period(CALENDAR_ORIGIN, freq="M")
    year_month = [(origin + int(value)).strftime("%Y-%m") for value in month]
    post = (month >= POLICY_START_MONTH).astype(int)
    return pd.DataFrame(
        {
            "month": month,
            "year_month": year_month,
            "post": post,
            "common_time": common_time_index(month, shock),
        }
    )
