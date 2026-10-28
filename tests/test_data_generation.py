import numpy as np
import pandas as pd
import pytest

from src.config import (
    IDENTIFICATION_NOTES,
    N_MONTHS,
    N_REGIONS,
    POLICY_START_MONTH,
    REGION_TYPES,
)
from src.data_generation import generate_panel, seasonal_component
from src.estimand import att_target_from_panel, design_target

# 处理地区等权：4*6 + 8*3 + 8*1，再除以 20 个处理地区。
HAND_TARGET = (4 * 6.0 + 8 * 3.0 + 8 * 1.0) / 20
UNWEIGHTED_TYPE_MEAN = (6.0 + 3.0 + 1.0) / 3
ALL_REGION_WEIGHTED = (18 * 6.0 + 18 * 3.0 + 12 * 1.0) / 48


def test_design_counts_and_hand_calculated_target():
    assert [(spec.name, spec.n_regions, spec.n_treated, spec.tau) for spec in REGION_TYPES] == [
        ("core", 18, 4, 6.0),
        ("growth", 18, 8, 3.0),
        ("emerging", 12, 8, 1.0),
    ]
    assert HAND_TARGET == pytest.approx(2.8)
    assert design_target(REGION_TYPES) == pytest.approx(HAND_TARGET)
    assert abs(HAND_TARGET - UNWEIGHTED_TYPE_MEAN) > 0.4
    assert abs(HAND_TARGET - ALL_REGION_WEIGHTED) > 0.4


def test_identification_notes_record_the_dgp():
    assert "平行趋势" in IDENTIFICATION_NOTES
    assert "季节" in IDENTIFICATION_NOTES
    assert "promo_intensity" in IDENTIFICATION_NOTES
    assert "主模型" in IDENTIFICATION_NOTES
    assert "2.8" in IDENTIFICATION_NOTES


def test_same_seed_is_reproducible():
    first = generate_panel(11)
    second = generate_panel(11)
    pd.testing.assert_frame_equal(first, second)


def test_panel_definition_and_treatment_composition():
    panel = generate_panel(3)
    assert list(panel.columns) == [
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
    ]
    assert len(panel) == N_REGIONS * N_MONTHS
    assert panel["region_id"].nunique() == N_REGIONS
    assert set(panel["month"]) == set(range(N_MONTHS))
    assert panel["year_month"].iloc[0] == "2022-01"
    assert panel.loc[panel["month"] == POLICY_START_MONTH, "year_month"].iloc[0] == "2023-01"
    assert panel["treated_post"].eq(panel["treated"] * panel["post"]).all()
    assert panel.loc[panel["month"] < POLICY_START_MONTH, "post"].eq(0).all()
    assert panel.loc[panel["month"] >= POLICY_START_MONTH, "post"].eq(1).all()
    assert panel.loc[panel["month"] < POLICY_START_MONTH, "treated_post"].eq(0).all()

    regions = panel.drop_duplicates("region_id")
    treated_counts = regions.groupby("region_type")["treated"].sum().to_dict()
    assert treated_counts == {"core": 4, "growth": 8, "emerging": 8}
    treated = regions.loc[regions["treated"] == 1, "treatment_start_month"]
    assert treated.notna().all()
    assert treated.eq(POLICY_START_MONTH).all()
    assert regions.loc[regions["treated"] == 0, "treatment_start_month"].isna().all()
    assert "tau" not in panel.columns
    assert att_target_from_panel(panel, {spec.name: spec.tau for spec in REGION_TYPES}) == pytest.approx(
        HAND_TARGET
    )


def test_structural_sample_keeps_assignment_and_removes_noise_dependence():
    structural = generate_panel(5, structural=True)
    noisy = generate_panel(5)
    pd.testing.assert_series_equal(structural["treated"], noisy["treated"])
    assert not np.allclose(structural["outcome"], noisy["outcome"])


def test_different_seeds_change_assignment():
    first = generate_panel(0, structural=True)
    second = generate_panel(1, structural=True)
    assert not first["treated"].equals(second["treated"])


def test_seasonality_is_common_and_repeats_each_year():
    month = np.arange(N_MONTHS)
    seasonal = seasonal_component(month)
    assert seasonal[0] == pytest.approx(seasonal[12])
    assert np.ptp(seasonal[:12]) > 1


def test_no_pretrend_and_no_treatment_effect_on_covariate_across_seeds():
    slopes = []
    promo_effects = []
    level_gaps = []
    for seed in range(6):
        panel = generate_panel(seed, structural=True)
        slopes.append(_pretrend_slope(panel))
        promo_effects.append(_two_by_two(panel, "promo_intensity"))
        level_gaps.append(_pre_level_gap(panel))
        assert _two_by_two(panel, "outcome") == pytest.approx(HAND_TARGET, abs=1e-8)
    assert max(abs(slope) for slope in slopes) < 1e-8
    assert max(abs(effect) for effect in promo_effects) < 1e-8
    assert min(abs(gap) for gap in level_gaps) > 5


def _two_by_two(df: pd.DataFrame, column: str) -> float:
    means = df.groupby(["treated", "post"])[column].mean()
    return float(means.loc[(1, 1)] - means.loc[(1, 0)] - (means.loc[(0, 1)] - means.loc[(0, 0)]))


def _pre_level_gap(df: pd.DataFrame) -> float:
    pre = df[df["month"] < POLICY_START_MONTH]
    means = pre.groupby("treated")["outcome"].mean()
    return float(means.loc[1] - means.loc[0])


def _pretrend_slope(df: pd.DataFrame) -> float:
    pre = df[df["month"] < POLICY_START_MONTH]
    means = pre.groupby(["month", "treated"], as_index=False)["outcome"].mean()
    wide = means.pivot(index="month", columns="treated", values="outcome").sort_index()
    gap = (wide[1] - wide[0]).to_numpy(dtype=float)
    x = wide.index.to_numpy(dtype=float)
    x = x - x.mean()
    return float(np.sum(x * (gap - gap.mean())) / np.sum(x**2))
