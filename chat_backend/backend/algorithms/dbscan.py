"""DBSCAN Clustering Algorithm"""

import time, numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from algochat_base import algorithm, Param, Input, Output
from algorithms._utils import (
    read_data, generate_fake_data, get_numeric_df, rv, save_plot, PALETTE, FIG_BG, PLOT_BG
)
import matplotlib.pyplot as plt


@algorithm(
    id="dbscan",
    name="DBSCAN 聚类",
    category="聚类",
    icon="🔵",
    desc="基于密度的聚类，自动发现簇数",
    inputs=[Input("data", types=["csv", "xlsx", "json"], desc="输入数据表")],
    outputs=[
        Output("scatter", type="chart", desc="聚类散点图"),
        Output("image", type="image", desc="matplotlib图"),
        Output("stats", type="table", desc="聚类统计"),
        Output("report", type="document", desc="分析报告"),
    ],
    params=[
        Param("eps", type="float", default=0.5, min=0.1, max=3.0, step=0.1, label="邻域半径ε"),
        Param("min_samples", type="int", default=5, min=2, max=50, step=1, label="最小样本数"),
    ],
)
def run_dbscan(inputs: dict, params: dict):
    eps = float(params.get("eps", 0.5))
    min_samples = int(params.get("min_samples", 5))

    files = inputs.get("data", [])
    numeric_df = get_numeric_df(read_data(files)) if files else generate_fake_data("clusters", 200)
    X = numeric_df.values
    if X.shape[1] < 2: X = np.column_stack([X, np.zeros(len(X))])
    Xs = StandardScaler().fit_transform(X)

    t0 = time.time()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
    elapsed = round(time.time() - t0, 3)
    nc = len(set(labels)) - (1 if -1 in labels else 0)
    nn = int((labels == -1).sum())

    datasets = []
    for i in range(nc):
        m = labels == i
        datasets.append({"label": f"聚类 {i+1}", "data": [{"x": rv(X[m,0][j]), "y": rv(X[m,1][j])} for j in range(m.sum())]})
    if nn > 0:
        m = labels == -1
        datasets.append({"label": "噪声点", "data": [{"x": rv(X[m,0][j]), "y": rv(X[m,1][j])} for j in range(m.sum())]})

    rows = [[str(i+1), str(int((labels==i).sum())), str(rv(X[labels==i,0].mean())), str(rv(X[labels==i,1].mean()))] for i in range(nc)]
    if nn > 0: rows.append(["噪声", str(nn), "-", "-"])

    sil = "N/A"
    if 1 < nc < len(X):
        try: sil = rv(silhouette_score(Xs[labels != -1], labels[labels != -1]))
        except: pass

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(PLOT_BG)
    for i in range(nc):
        m = labels == i
        ax.scatter(X[m, 0], X[m, 1], c=PALETTE[i % len(PALETTE)], label=f'聚类 {i+1}', alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    if nn > 0:
        m = labels == -1
        ax.scatter(X[m, 0], X[m, 1], c='#999999', marker='x', label='噪声点', alpha=0.5, s=20)
    ax.set_title('DBSCAN 聚类结果', fontsize=14, fontweight='bold', color='#3C2819')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    img = save_plot(fig, "dbscan_scatter")

    return {
        "scatter": {"chartType": "scatter", "datasets": datasets},
        "image": img,
        "stats": {"columns": ["聚类", "样本数", "中心X", "中心Y"], "rows": rows},
        "report": f"算法: DBSCAN\n参数: eps={eps}, min_samples={min_samples}\n时间: {elapsed}s\n聚类数: {nc}\n噪声: {nn}\n轮廓系数: {sil}",
    }