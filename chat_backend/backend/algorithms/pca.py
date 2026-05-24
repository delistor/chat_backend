"""PCA Dimensionality Reduction Algorithm"""

import time, numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from algochat_base import algorithm, Param, Input, Output
from algorithms._utils import (
    read_data, generate_fake_data, get_numeric_df, rv, save_plot, PALETTE, FIG_BG, PLOT_BG
)
import matplotlib.pyplot as plt


@algorithm(
    id="pca",
    name="PCA 降维",
    category="降维",
    icon="🔄",
    desc="主成分分析降维，提取数据主要特征",
    inputs=[Input("data", types=["csv", "xlsx", "json"], desc="输入数据表")],
    outputs=[
        Output("scatter", type="chart", desc="PCA散点图"),
        Output("image", type="image", desc="matplotlib图"),
        Output("variance", type="chart", desc="方差解释比"),
        Output("components", type="table", desc="主成分表"),
        Output("report", type="document", desc="分析报告"),
    ],
    params=[
        Param("n_components", type="int", default=2, min=1, max=10, step=1, label="目标维度"),
    ],
)
def run_pca(inputs: dict, params: dict):
    nc = int(params.get("n_components", 2))
    files = inputs.get("data", [])
    numeric_df = get_numeric_df(read_data(files)) if files else generate_fake_data("multivariate", 200)
    X = numeric_df.values
    if nc > X.shape[1]: nc = X.shape[1]
    Xs = StandardScaler().fit_transform(X)

    t0 = time.time()
    pca_model = PCA(n_components=nc)
    Xp = pca_model.fit_transform(Xs)
    exp = pca_model.explained_variance_ratio_
    elapsed = round(time.time() - t0, 3)
    cum = np.cumsum(exp)

    if nc >= 2:
        datasets = [{"label": "PCA结果", "data": [{"x": rv(Xp[i,0]), "y": rv(Xp[i,1])} for i in range(len(Xp))]}]
    else:
        datasets = [{"label": "PC1", "data": [{"x": i, "y": rv(Xp[i,0])} for i in range(len(Xp))]}]

    rows = [[f"PC{i+1}", str(rv(exp[i])), str(rv(exp[i]*100,2))+"%", str(rv(cum[i]*100,2))+"%"] for i in range(nc)]
    var_chart = {"labels": [f"PC{i+1}" for i in range(nc)], "datasets": [
        {"label": "方差解释比", "data": [rv(v*100,2) for v in exp]},
        {"label": "累计方差", "data": [rv(v*100,2) for v in cum]}]}

    # Matplotlib figure
    img = None
    if nc >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.patch.set_facecolor(FIG_BG)
        ax = axes[0]; ax.set_facecolor(PLOT_BG)
        ax.scatter(Xp[:, 0], Xp[:, 1], c=PALETTE[0], alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
        ax.set_title('PCA 降维结果', fontsize=14, fontweight='bold', color='#3C2819')
        ax.set_xlabel(f'PC1 ({rv(exp[0]*100,1)}%)', color='#64503C')
        ax.set_ylabel(f'PC2 ({rv(exp[1]*100,1)}%)', color='#64503C')
        ax.grid(True, alpha=0.2)
        ax2 = axes[1]; ax2.set_facecolor(PLOT_BG)
        x_pos = range(1, nc + 1)
        ax2.bar(x_pos, [v*100 for v in exp], color=PALETTE[:nc], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.plot(x_pos, [v*100 for v in cum], 'o-', color=PALETTE[1], linewidth=2, markersize=6, label='累计方差')
        ax2.set_title('方差解释比', fontsize=14, fontweight='bold', color='#3C2819')
        ax2.set_ylabel('百分比 (%)', color='#64503C')
        ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        img = save_plot(fig, "pca_combined")

    return {
        "scatter": {"chartType": "scatter", "datasets": datasets},
        "image": img or {"src": "", "name": ""},
        "variance": {"chartType": "bar", **var_chart},
        "components": {"columns": ["主成分", "特征值", "方差解释比", "累计解释率"], "rows": rows},
        "report": f"算法: PCA\n原始: {X.shape[1]}维 → {nc}维\n样本: {len(X)}\n累计解释率: {rv(cum[-1]*100,2)}%\n时间: {elapsed}s",
    }