"""
Shared utilities for algorithm modules.
"""

import os, uuid, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMAGE_DIR = os.path.join(tempfile.gettempdir(), "algochat_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PALETTE = ['#7D9E8D', '#8E4E26', '#992D1E', '#A0B9AA', '#C49B60', '#5F806F', '#AF7850', '#BE5A46']

FIG_BG = '#FAFCF5'
PLOT_BG = '#FFFFFF'


def read_data(files):
    if not files: return None
    path = files[0]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv": return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"): return pd.read_excel(path)
    elif ext == ".json": return pd.read_json(path)
    else: raise ValueError(f"不支持的文件格式: {ext}")


def generate_fake_data(dataset_type="clusters", n_samples=200):
    np.random.seed(42)
    if dataset_type == "clusters":
        centers = [[2,2],[-2,-2],[3,-3]]
        data = [np.random.randn(n_samples//3, 2)*0.8 + c for c in centers]
        return pd.DataFrame(np.vstack(data), columns=["x","y"])
    elif dataset_type == "regression":
        X = np.random.uniform(-5, 5, (n_samples, 1))
        y = 2.5*X.ravel()+3.0+np.random.randn(n_samples)*1.5
        return pd.DataFrame({"x": X.ravel(), "y": y})
    elif dataset_type == "multivariate":
        cols = [f"feature_{i+1}" for i in range(5)]
        return pd.DataFrame(np.random.randn(n_samples, 5), columns=cols)
    elif dataset_type == "anomalies":
        n_normal = int(n_samples*0.9)
        normal = np.random.randn(n_normal, 2)*0.5
        anomalies = np.random.uniform(-6, 6, (n_samples-n_normal, 2))
        return pd.DataFrame(np.vstack([normal, anomalies]), columns=["x","y"])
    return pd.DataFrame(np.random.randn(n_samples, 2), columns=["x","y"])


def get_numeric_df(df):
    ndf = df.select_dtypes(include=[np.number])
    if ndf.shape[1] < 1: raise ValueError("数据中未找到数值列")
    return ndf


def rv(v, d=4):
    if isinstance(v, (float, np.floating)): return round(float(v), d)
    return v


def save_plot(fig, name_prefix):
    file_id = f"img_{uuid.uuid4().hex[:8]}"
    filename = f"{name_prefix}_{file_id}.png"
    filepath = os.path.join(IMAGE_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=FIG_BG, edgecolor='none')
    plt.close(fig)
    url = f"/api/images/{filename}"
    return {"src": url, "name": f"{name_prefix}.png"}