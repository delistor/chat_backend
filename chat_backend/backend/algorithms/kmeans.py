"""KMeans Clustering Algorithm"""

import time, numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from algochat_base import algorithm, Param, Input, Output
from algorithms._utils import (
    read_data, generate_fake_data, get_numeric_df, rv, save_plot, PALETTE, FIG_BG, PLOT_BG
)
import matplotlib.pyplot as plt


@algorithm(
    id="kmeans",
    name="KMeans 聚类",
    category="聚类",
    icon="🎯",
    desc="K-Means聚类分析，将数据分为K个簇",
    inputs=[Input("data", types=["csv", "xlsx", "json"], desc="输入数据表")],
    outputs=[
        Output("scatter", type="chart", desc="聚类散点图"),
        Output("image", type="image", desc="matplotlib散点图"),
        Output("stats", type="table", desc="聚类统计结果"),
        Output("report", type="document", desc="分析报告"),
    ],
    params=[
        Param("k", type="int", default=3, min=2, max=20, step=1, label="聚类数K"),
        Param("max_iter", type="int", default=300, min=50, max=1000, step=50, label="最大迭代"),
    ],
)
def run_kmeans(inputs: dict, params: dict):
    k = int(params.get("k", 3))
    max_iter = int(params.get("max_iter", 300))

    files = inputs.get("data", [])
    numeric_df = get_numeric_df(read_data(files)) if files else generate_fake_data("clusters", 200)
    X = numeric_df.values
    if X.shape[1] < 2:
        X = np.column_stack([X, np.zeros(len(X))])
    Xs = StandardScaler().fit_transform(X)

    t0 = time.time()
    labels = KMeans(n_clusters=k, max_iter=max_iter, n_init=10, random_state=42).fit_predict(Xs)
    elapsed = round(time.time() - t0, 3)

    # Chart data
    datasets = []
    for i in range(k):
        m = labels == i
        datasets.append({
            "label": f"聚类 {i+1}",
            "data": [{"x": rv(X[m, 0][j]), "y": rv(X[m, 1][j])} for j in range(m.sum())]
        })

    # Table data
    rows = [
        [str(i+1), str(int((labels==i).sum())),
         str(rv(X[labels==i, 0].mean())), str(rv(X[labels==i, 1].mean())),
         str(rv(X[labels==i].var()))]
        for i in range(k)
    ]
    sil = rv(silhouette_score(Xs, labels)) if 1 < k < len(X) else "N/A"

    # Matplotlib image
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PLOT_BG)
    for i in range(k):
        m = labels == i
        ax.scatter(X[m, 0], X[m, 1], c=PALETTE[i % len(PALETTE)],
                   label=f'聚类 {i+1}', alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    centroids = np.array([X[labels==i].mean(axis=0) for i in range(k)])
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200,
               edgecolors='white', linewidth=2, zorder=5, label='中心')
    ax.set_title('KMeans 聚类结果', fontsize=14, fontweight='bold', color='#3C2819')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    img = save_plot(fig, "kmeans_scatter")

    return {
        "scatter": {"chartType": "scatter", "datasets": datasets},
        "image": img,
        "stats": {"columns": ["聚类", "样本数", "中心X", "中心Y", "方差"], "rows": rows},
        "report": f"算法: KMeans\n参数: k={k}, max_iter={max_iter}\n时间: {elapsed}s\n样本: {len(X)}\n轮廓系数: {sil}",
    }