"""Linear Regression Algorithm"""

import time, numpy as np
from sklearn.linear_model import LinearRegression

from algochat_base import algorithm, Param, Input, Output
from algorithms._utils import (
    read_data, generate_fake_data, get_numeric_df, rv, save_plot, PALETTE, FIG_BG, PLOT_BG
)
import matplotlib.pyplot as plt


@algorithm(
    id="linear_regression",
    name="线性回归",
    category="回归",
    icon="📈",
    desc="线性回归分析，拟合数据并评估模型",
    inputs=[Input("data", types=["csv", "xlsx", "json"], desc="输入数据表")],
    outputs=[
        Output("chart", type="chart", desc="回归拟合图"),
        Output("image", type="image", desc="matplotlib图"),
        Output("coefficients", type="table", desc="回归系数"),
        Output("report", type="document", desc="分析报告"),
    ],
    params=[],
)
def run_linear_regression(inputs: dict, params: dict):
    files = inputs.get("data", [])
    numeric_df = get_numeric_df(read_data(files)) if files else generate_fake_data("regression", 200)
    X = numeric_df.iloc[:, :-1].values
    y = numeric_df.iloc[:, -1].values
    if X.shape[1] < 1: raise ValueError("至少需要2列数值数据")

    t0 = time.time()
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    elapsed = round(time.time() - t0, 3)
    r2 = rv(model.score(X, y))
    mse = rv(np.mean((y - y_pred) ** 2))
    coefs = [rv(c) for c in model.coef_]
    intercept = rv(model.intercept_)

    if X.shape[1] == 1:
        si = X[:, 0].argsort()
        chart_data = {"labels": [str(rv(v, 2)) for v in X[si, 0]], "datasets": [
            {"label": "实际值", "data": [rv(y[i]) for i in si]},
            {"label": "拟合线", "data": [rv(y_pred[i]) for i in si]}]}
        ct = "line"
    else:
        chart_data = {"datasets": [{"label": "实际vs预测", "data": [{"x": rv(y[i]), "y": rv(y_pred[i])} for i in range(len(y))]}]}
        ct = "scatter"

    rows = [[numeric_df.columns[i], str(coefs[i])] for i in range(len(coefs))] + [["截距", str(intercept)]]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(PLOT_BG)
    if X.shape[1] == 1:
        si = X[:, 0].argsort()
        ax.scatter(X, y, c=PALETTE[0], alpha=0.5, s=25, label='实际值', edgecolors='white', linewidth=0.3)
        ax.plot(X[si], y_pred[si], color=PALETTE[1], linewidth=2.5, label='拟合线')
    else:
        ax.scatter(y, y_pred, c=PALETTE[0], alpha=0.5, s=25, edgecolors='white', linewidth=0.3)
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax.plot(lims, lims, '--', color=PALETTE[1], linewidth=2, label='理想线')
        ax.set_xlabel('实际值', color='#64503C'); ax.set_ylabel('预测值', color='#64503C')
    ax.set_title(f'线性回归 (R²={r2})', fontsize=14, fontweight='bold', color='#3C2819')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    img = save_plot(fig, "linear_regression")

    return {
        "chart": {"chartType": ct, "datasets": chart_data.get("datasets"), **({"labels": chart_data["labels"]} if "labels" in chart_data else {})},
        "image": img,
        "coefficients": {"columns": ["特征", "系数"], "rows": rows},
        "report": f"算法: 线性回归\n特征数: {X.shape[1]}\n样本: {len(X)}\nR²: {r2}\nMSE: {mse}\n截距: {intercept}\n时间: {elapsed}s",
    }