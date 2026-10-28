"""展示用图形。图只读取已经算好的结果或原始面板，不重新拟合模型。"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, fontManager

from src.event_study import EventStudyResult


def plot_event_study(result: EventStudyResult, ax: Axes | None = None) -> Axes:
    """绘制动态系数和 95% 置信区间。基准期单独标记，不画成一个估计为零的系数。"""

    estimated = [item for item in result.estimates if item.estimated]
    if not estimated:
        raise ValueError("事件研究结果里没有已估计的相对时期，无法绘图。")
    reference = next(item for item in result.estimates if item.role == "reference")
    font = _available_font(("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"))
    context = {"axes.unicode_minus": False}
    if font is not None:
        context["font.sans-serif"] = [font, "DejaVu Sans"]
    with plt.rc_context(context):
        if ax is None:
            _figure, ax = plt.subplots(figsize=(10, 5))
        x_values = [item.event_time for item in estimated]
        y_values = [float(item.estimate) for item in estimated]
        y_error = np.vstack(
            [
                [float(item.estimate) - float(item.ci_low) for item in estimated],
                [float(item.ci_high) - float(item.estimate) for item in estimated],
            ]
        )
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_error,
            fmt="o",
            color="#1f4e79",
            ecolor="#5b7c99",
            elinewidth=1.2,
            capsize=3,
            label="已估计系数及 95% 置信区间",
        )
        ax.scatter(
            [reference.event_time],
            [0.0],
            marker="D",
            s=35,
            color="#b85c38",
            label="基准期（省略约束，不是估计值）",
            zorder=3,
        )
        ax.axhline(0.0, color="#666666", linewidth=1, label="零效应")
        ax.axvline(-0.5, color="#9a3412", linewidth=1, linestyle="--", label="策略开始（相对时间 0）")
        ax.set_title("事件研究：统一处理时点（模拟数据）")
        ax.set_xlabel("相对处理时间（月）")
        ax.set_ylabel("动态处理效应估计")
        ax.set_xticks(_tick_times(result.estimates))
        ax.legend(frameon=False)
        ax.figure.text(
            0.01,
            0.01,
            "竖线区间是地区聚类标准误下的 95% 置信区间。菱形是被省略的基准期，不是估计为零。"
            "处理前系数不显著不能证明平行趋势。本图只对应统一处理时点。",
            fontsize=8,
        )
        ax.figure.subplots_adjust(bottom=0.18)
        _apply_cjk(ax)
    return ax


def _tick_times(estimates: Sequence[object]) -> list[int]:
    times = [int(item.event_time) for item in estimates]
    return list(range(min(times), max(times) + 1, 2))


def plot_outcome_trends(df, policy_start: int | None, ax: Axes | None = None) -> Axes:
    """绘制处理组与对照组的月度平均结果。这是描述统计，不重新估计模型。"""

    means = df.groupby(["month", "treated"], as_index=False)["outcome"].mean()
    with _cn_style():
        ax = _ensure_ax(ax, (9, 4.8))
        for treated, label, color in ((0, "对照组", "#5b7c99"), (1, "处理组", "#1f4e79")):
            subset = means.loc[means["treated"] == treated].sort_values("month")
            ax.plot(subset["month"], subset["outcome"], marker="o", label=label, color=color)
        if policy_start is not None:
            ax.axvline(policy_start - 0.5, color="#9a3412", linestyle="--", label="策略开始")
        ax.set_title("处理组与对照组的平均结果（模拟数据）")
        ax.set_xlabel("月份")
        ax.set_ylabel("平均单位用户收益")
        ax.legend(frameon=False)
        _apply_cjk(ax)
    return ax


def plot_interval_estimates(
    labels: Sequence[str],
    estimates: Sequence[float],
    ci_low: Sequence[float],
    ci_high: Sequence[float],
    *,
    title: str,
    xlabel: str,
    ax: Axes | None = None,
) -> Axes:
    """用已经算好的系数和区间画横向误差线图。"""

    if not labels:
        raise ValueError("没有可绘制的估计值。")
    y_pos = np.arange(len(labels))
    estimates_array = np.asarray(estimates, dtype=float)
    low = estimates_array - np.asarray(ci_low, dtype=float)
    high = np.asarray(ci_high, dtype=float) - estimates_array
    with _cn_style():
        ax = _ensure_ax(ax, (9, max(3.5, 0.55 * len(labels) + 1.5)))
        ax.errorbar(
            estimates_array,
            y_pos,
            xerr=np.vstack([low, high]),
            fmt="o",
            color="#1f4e79",
            ecolor="#5b7c99",
            capsize=3,
            label="估计值及 95% 置信区间",
        )
        ax.axvline(0.0, color="#666666", linewidth=1, label="零效应")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(labels))
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        _apply_cjk(ax)
    return ax


def plot_placebo_distribution(estimates: Sequence[float], actual: float, ax: Axes | None = None) -> Axes:
    """比较真实估计和虚假处理组的模拟分布。"""

    if len(estimates) == 0:
        raise ValueError("没有虚假处理组估计。")
    with _cn_style():
        ax = _ensure_ax(ax, (9, 4.5))
        ax.hist(list(estimates), bins=min(12, max(5, len(estimates) // 2)), color="#5b7c99", edgecolor="white")
        ax.axvline(actual, color="#9a3412", linewidth=2, label="真实处理分配的估计")
        ax.set_title("虚假处理组估计的模拟分布（模拟数据）")
        ax.set_xlabel("双重差分估计")
        ax.set_ylabel("重复次数")
        ax.legend(frameon=False)
        _apply_cjk(ax)
    return ax


def _ensure_ax(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    if ax is None:
        _figure, ax = plt.subplots(figsize=figsize)
    return ax


def _cn_style():
    font = _available_font(("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"))
    context = {"axes.unicode_minus": False}
    if font is not None:
        context["font.sans-serif"] = [font, "DejaVu Sans"]
    return plt.rc_context(context)


def _apply_cjk(ax: Axes) -> None:
    """把已经创建的文字绑到字体文件上。Streamlit 在函数返回后才渲染，不能只依赖临时 rc。"""

    font = _cjk_properties()
    if font is None:
        return
    texts = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
    legend = ax.get_legend()
    if legend is not None:
        texts.extend(legend.get_texts())
    texts.extend(ax.figure.texts)
    for text in texts:
        text.set_fontproperties(font)


def _cjk_properties() -> FontProperties | None:
    names = {font.name: font.fname for font in fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        path = names.get(candidate)
        if path:
            return FontProperties(fname=path)
    return None


def _available_font(candidates: tuple[str, ...]) -> str | None:
    names = {font.name for font in fontManager.ttflist}
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None
