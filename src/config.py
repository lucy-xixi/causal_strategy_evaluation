"""模拟设计、真实效应和识别假设。

基础双重差分的目标不是各 region_type 处理效应的简单平均。
在平衡面板且所有处理地区于同一月份开始接受处理时，地区固定效应和时间固定效应
回归中的 treated_post 系数，对应处理地区等权平均处理效应：

    tau_ATT = (4 * 6 + 8 * 3 + 8 * 1) / 20 = 2.8

其中 4、8、8 是 core、growth、emerging 的处理地区数。
三种类型效应的简单平均是 (6 + 3 + 1) / 3 = 3.333...
按全部地区数加权是 (18 * 6 + 18 * 3 + 12 * 1) / 48 = 3.625。
后两个数都不是本设计的估计目标。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionTypeSpec:
    """一种地区类型的样本量、处理分配和静态处理效应。"""

    name: str
    n_regions: int
    n_treated: int
    tau: float
    baseline_outcome: float
    promo_level: float


# 固定随机种子。同一种子下，处理分配和噪声都可复现。
DEFAULT_SEED = 42

N_REGIONS = 48
N_MONTHS = 24
MONTH_MIN = 0
MONTH_MAX = N_MONTHS - 1
# 月份 0–11 为处理前，12–23 为处理后。处理地区都在这个月份开始。
POLICY_START_MONTH = 12
CALENDAR_ORIGIN = "2022-01"

# 处理地区数故意不与类型规模成比例，避免估计目标退化为类型效应的简单平均。
# 每个类型都保留对照地区，便于后续做分层估计；本阶段只估计汇总 ATT。
REGION_TYPES: tuple[RegionTypeSpec, ...] = (
    RegionTypeSpec("core", 18, 4, 6.0, 70.0, 1.2),
    RegionTypeSpec("growth", 18, 8, 3.0, 50.0, 0.8),
    RegionTypeSpec("emerging", 12, 8, 1.0, 32.0, 0.4),
)

TREND_PER_MONTH = 0.20
SEASON_SIN = 2.5
SEASON_COS = 1.2
PROMO_TREND_PER_MONTH = 0.05
PROMO_SEASON_SIN = 0.3
# promo_intensity 进入结果方程，但不进入基础 DiD 的控制变量。
PROMO_COEF_IN_OUTCOME = 0.40

OUTCOME_NOISE_SD = 2.0
COVARIATE_NOISE_SD = 0.5
COMMON_SHOCK_SD = 1.0
REGION_INTERCEPT_SD = 4.0
REGION_BLOCK_SHOCK_SD = 1.0
BLOCK_LENGTH = 6
PROMO_REGION_NOISE_SD = 0.3

CI_ALPHA = 0.05
# 常规聚类标准误的大样本依据需要较多聚类。低于该数量时另外给出更强警告。
CLUSTER_CAUTION_BELOW = 30

IDENTIFICATION_NOTES = """
本数据是模拟面板，不是企业真实成效。

1. 处理分配
   地区先按 region_type 固定数量生成，再在类型内部用固定种子无放回抽取处理地区。
   core、growth、emerging 的处理地区数分别为 4、8、8。
   处理只改变实施后的结果水平，不改变地区被分配到处理组的规则。
   所有处理地区的 treatment_start_month 都是第 12 个月，没有交错采纳，也没有预期效应。

2. 平行趋势
   处理前不设置系统性的组间差异趋势。地区基线 region_baseline 不随月份变化。
   处理组和对照组可以有不同的平均水平，因为不同类型的基线不同，且类型之间的处理比例不同。
   这种水平差异由地区固定效应吸收，不是差异趋势。

3. 共同时间冲击与季节性
   所有地区共享同一个时间指数：
   common_time(month) = 0.20 * month
       + 2.5 * sin(2π * (month % 12) / 12)
       + 1.2 * cos(2π * (month % 12) / 12)
       + 该月的共同冲击。
   季节项只依赖月份在一年中的位置，因此第 0 个月与第 12 个月的季节项相同。
   共同冲击对所有地区相同，按月抽取一次。月份固定效应吸收趋势、季节性和共同冲击。

4. 协变量
   region_baseline 是时间不变的地区水平，会因地区类型而不同，因此与处理分配相关。
   它不被策略改变，也不进入主模型；地区固定效应已经吸收它。
   promo_intensity 的生成不使用 treated、post 或 treated_post：
   promo = 类型水平 + 地区噪声 + 0.05 * month + 0.3 * sin(2π * (month % 12) / 12) + 独立噪声。
   它会影响 outcome，但不是策略造成的。主模型故意不控制它。
   它的地区水平和共同时间部分会被两类固定效应吸收，独立噪声与处理正交，
   所以省略它不改变当前的 ATT。本设计不把受策略影响的变量放进主模型。

5. 结果方程
   outcome = region_baseline + common_time + 0.40 * promo_intensity
       + tau(region_type) * treated_post
       + 地区分段冲击 + 独立噪声。
   tau 在实施后的各个月份保持不变，因此静态双重差分不会在事件时间之间再加权。
   基础模型的总体目标是处理地区等权 ATT = 2.8。

6. 误差与聚类
   地区分段冲击在同一地区的相邻月份中相关，所以报告地区聚类标准误。
   聚类数等于地区数 48。这仍是有限聚类，常规聚类稳健协方差可能低估不确定性。
""".strip()


def tau_by_type() -> dict[str, float]:
    """返回地区类型到静态处理效应的映射。估计函数默认不知道这个真值。"""

    return {spec.name: spec.tau for spec in REGION_TYPES}


def validate_design() -> None:
    """检查配置内部是否能支持统一时点的双重差分。"""

    total_regions = sum(spec.n_regions for spec in REGION_TYPES)
    if total_regions != N_REGIONS:
        raise ValueError("REGION_TYPES 的地区数之和必须等于 N_REGIONS。")
    for spec in REGION_TYPES:
        if spec.n_regions <= 0:
            raise ValueError(f"{spec.name} 的地区数必须为正。")
        if not 0 < spec.n_treated < spec.n_regions:
            raise ValueError(f"{spec.name} 必须同时有处理地区和对照地区。")
    if not MONTH_MIN <= POLICY_START_MONTH <= MONTH_MAX:
        raise ValueError("策略开始月份必须落在样本月份范围内。")
    if POLICY_START_MONTH == MONTH_MIN:
        raise ValueError("策略开始月份不能是第一个月，否则没有处理前时期。")
    if N_MONTHS % BLOCK_LENGTH != 0:
        raise ValueError("月份数必须能被分段冲击长度整除。")
