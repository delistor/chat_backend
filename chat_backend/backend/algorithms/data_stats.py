"""Descriptive Statistics Algorithm"""

import time, numpy as np

from algochat_base import algorithm, Param, Input, Output
from algorithms._utils import (
    read_data, generate_fake_data, get_numeric_df, rv, save_plot, PALETTE, FIG_BG, PLOT_BG
)
import matplotlib.pyplot as plt


@algorithm(
    id="data_stats",
    name="数据统计",
    category="统计",
    icon="📊",
    desc="描述性统计分析，计算均值、标准差等",
    inputs=[Input("data", types=["csv", "xlsx", "json"], desc="输入数据表")],
    outputs=[
        Output("chart", type="chart", desc="均值/标准差图"),
        Output("image", type="image", desc="matplotlib图"),
        Output("table", type="table", desc="描述统计表"),
        Output("report", type="document", desc="分析报告"),
    ],
    params=[],
)
def run_stats(inputs: dict, params: dict):
    files = inputs.get("data", [])
    numeric_df = get_numeric_df(read_data(files)) if files else generate_fake_data("multivariate", 200)

    t0 = time.time()
    desc = numeric_df.describe().T
    elapsed = round(time.time() - t0, 3)

    columns = ["字段", "计数", "均值", "标准差", "最小值", "25%", "中位数", "75%", "最大值"]
    rows = [[str(col), str(int(desc.loc[col, "count"])), str(rv(desc.loc[col, "mean"])),
             str(rv(desc.loc[col, "std"])), str(rv(desc.loc[col, "min"])),
             str(rv(desc.loc[col, "25%"])), str(rv(desc.loc[col, "50%"])),
             str(rv(desc.loc[col, "75%"])), str(rv(desc.loc[col, "max"]))] for col in desc.index]

    chart_data = {"labels": list(desc.index), "datasets": [
        {"label": "均值", "data": [rv(desc.loc[c, "mean"]) for c in desc.index]},
        {"label": "标准差", "data": [rv(desc.loc[c, "std"]) for c in desc.index]}]}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(FIG_BG)
    ax = axes[0]; ax.set_facecolor(PLOT_BG)
    x_pos = range(len(desc.index)); width = 0.35
    ax.bar([p - width/2 for p in x_pos], [desc.loc[c, "mean"] for c in desc.index], width, color=PALETTE[0], alpha=0.8, label='均值', edgecolor='white')
    ax.bar([p + width/2 for p in x_pos], [desc.loc[c, "std"] for c in desc.index], width, color=PALETTE[1], alpha=0.8, label='标准差', edgecolor='white')
    ax.set_xticks(list(x_pos)); ax.set_xticklabels(desc.index, rotation=30, ha='right', fontsize=10)
    ax.set_title('均值 & 标准差', fontsize=14, fontweight='bold', color='#3C2819')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

    ax2 = axes[1]; ax2.set_facecolor(PLOT_BG)
    bp = ax2.boxplot([numeric_df[c].values for c in desc.index], tick_labels=desc.index, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(PALETTE[i % len(PALETTE)]); patch.set_alpha(0.6)
    ax2.set_title('数据分布箱线图', fontsize=14, fontweight='bold', color='#3C2819')
    ax2.tick_params(axis='x', rotation=30); ax2.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    img = save_plot(fig, "data_stats")

    return {
        "chart": {"chartType": "bar", **chart_data},
        "image": img,
        "table": {"columns": columns, "rows": rows},
        "report": f"算法: 描述统计\n字段: {numeric_df.shape[1]}\n样本: {numeric_df.shape[0]}\n时间: {elapsed}s",
    }