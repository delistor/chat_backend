"""Anomaly Detection Algorithm (Isolation Forest)"""

import time, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from algochat_base import algorithm, Param, Input, Output
from algorithms._utils import (
    read_data, generate_fake_data, get_numeric_df, rv, save_plot, PALETTE, FIG_BG, PLOT_BG
)
import matplotlib.pyplot as plt


@algorithm(
    id="anomaly_detection",
    name="异常检测",
    category="检测",
    icon="🔍",
    desc="基于孤立森林的异常点检测",
    inputs=[Input("data", types=["csv", "xlsx", "json"], desc="输入数据表")],
    outputs=[
        Output("scatter", type="chart", desc="异常检测图"),
        Output("image", type="image", desc="matplotlib图"),
        Output("stats", type="table", desc="异常统计"),
        Output("report", type="document", desc="分析报告"),
    ],
    params=[
        Param("contamination", type="float", default=0.1, min=0.01, max=0.5, step=0.01, label="污染率"),
    ],
)
def run_anomaly(inputs: dict, params: dict):
    contam = float(params.get("contamination", 0.1))
    files = inputs.get("data", [])
    numeric_df = get_numeric_df(read_data(files)) if files else generate_fake_data("anomalies", 200)
    X = numeric_df.values
    if X.shape[1] < 2: X = np.column_stack([X, np.zeros(len(X))])
    Xs = StandardScaler().fit_transform(X)

    t0 = time.time()
    labels = IsolationForest(contamination=contam, random_state=42).fit_predict(Xs)
    elapsed = round(time.time() - t0, 3)
    na = int((labels == -1).sum())
    nn = int((labels == 1).sum())

    datasets = [
        {"label": "正常点", "data": [{"x": rv(X[labels==1,0][j]), "y": rv(X[labels==1,1][j])} for j in range(nn)]},
        {"label": "异常点", "data": [{"x": rv(X[labels==-1,0][j]), "y": rv(X[labels==-1,1][j])} for j in range(na)]}
    ]
    rows = [["正常", str(nn), str(rv(nn/len(X)*100,2))+"%"], ["异常", str(na), str(rv(na/len(X)*100,2))+"%"], ["总计", str(len(X)), "100%"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(PLOT_BG)
    ax.scatter(X[labels==1, 0], X[labels==1, 1], c=PALETTE[0], alpha=0.6, s=30, label='正常点', edgecolors='white', linewidth=0.3)
    ax.scatter(X[labels==-1, 0], X[labels==-1, 1], c=PALETTE[2], alpha=0.8, s=50, marker='D', label='异常点', edgecolors='white', linewidth=0.5)
    ax.set_title(f'异常检测 (异常率 {rv(na/len(X)*100,1)}%)', fontsize=14, fontweight='bold', color='#3C2819')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    img = save_plot(fig, "anomaly_detection")

    return {
        "scatter": {"chartType": "scatter", "datasets": datasets},
        "image": img,
        "stats": {"columns": ["类别", "数量", "占比"], "rows": rows},
        "report": f"算法: 孤立森林\n污染率: {contam}\n样本: {len(X)}\n异常: {na}\n正常: {nn}\n时间: {elapsed}s",
    }