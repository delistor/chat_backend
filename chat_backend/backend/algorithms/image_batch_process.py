"""Image Batch Processing — Demo algorithm for grouped results.

Accepts multiple images (or generates fake ones), processes each,
and returns per-image results (image + table) grouped by input filename.
Optionally includes a trend chart across all images.
"""

import time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from algochat_base import algorithm, Param, Input, Output
from algorithms._utils import save_plot, PALETTE, FIG_BG, PLOT_BG

# Fake image names for demo (when no files uploaded)
FAKE_IMAGES = ["样品_A.png", "样品_B.png", "样品_C.png", "样品_D.png", "样品_E.png"]


def _fake_process_image(index: int, seed: int):
    """Simulate processing one image: generate a result image + detection values."""
    np.random.seed(seed)
    
    # Fake detection metrics
    score = round(np.random.uniform(0.6, 0.99), 4)
    confidence = round(np.random.uniform(0.7, 0.98), 4)
    anomaly_count = int(np.random.randint(0, 8))
    avg_intensity = round(np.random.uniform(100, 250), 2)
    snr = round(np.random.uniform(10, 45), 2)
    
    # Generate a fake result image (colored scatter pattern)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PLOT_BG)
    
    n_points = np.random.randint(30, 80)
    x = np.random.randn(n_points) * 2 + index
    y = np.random.randn(n_points) * 1.5
    ax.scatter(x, y, c=[PALETTE[c % len(PALETTE)] for c in np.random.randint(0, 5, n_points)],
               alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
    ax.set_title(f'检测结果 — 样品 {chr(65+index)}', fontsize=12, fontweight='bold', color='#3C2819')
    ax.grid(True, alpha=0.2)
    img_result = save_plot(fig, f"batch_img_{index}")
    
    return {
        "score": score,
        "confidence": confidence,
        "anomaly_count": anomaly_count,
        "avg_intensity": avg_intensity,
        "snr": snr,
        "image": img_result,
    }


@algorithm(
    id="image_batch",
    name="图片批处理",
    category="图像处理",
    icon="🖼",
    desc="批量处理多张图片，每张返回处理结果图和检测数值，可选趋势图",
    inputs=[
        Input("images", types=["png", "jpg", "jpeg", "bmp", "tiff"], multiple=True, desc="输入图片（可多选）"),
    ],
    outputs=[
        # Static outputs are not used — we use dynamic _results mode
    ],
    params=[
        Param("show_trend", type="select", default="是", options=["是", "否"], label="显示趋势图"),
        Param("threshold", type="float", default=0.85, min=0.5, max=1.0, step=0.05, label="检测阈值"),
        Param("mode", type="select", default="标准", options=["快速", "标准", "精细"], label="处理模式"),
    ],
)
def run_image_batch(inputs: dict, params: dict):
    show_trend = params.get("show_trend", "是") == "是"
    threshold = float(params.get("threshold", 0.85))
    mode = params.get("mode", "标准")
    
    # Get file names (or use fake ones)
    files = inputs.get("images", [])
    if files:
        import os
        image_names = [os.path.basename(f) for f in files]
    else:
        image_names = FAKE_IMAGES
    
    # Process each image
    per_image_results = []
    trend_scores = []
    trend_confidences = []
    trend_labels = []
    
    for i, name in enumerate(image_names):
        result = _fake_process_image(i, seed=42 + i * 7)
        
        trend_labels.append(name.replace('.png', '').replace('.jpg', ''))
        trend_scores.append(result["score"])
        trend_confidences.append(result["confidence"])
        
        # ── Per-image result: image (grouped) ──
        per_image_results.append({
            "name": "处理结果",
            "type": "image",
            "group": name,
            "src": result["image"]["src"],
        })
        
        # ── Per-image result: detection table (grouped) ──
        per_image_results.append({
            "name": "检测数值",
            "type": "table",
            "group": name,
            "columns": ["指标", "值"],
            "rows": [
                ["检测分数", str(result["score"])],
                ["置信度", str(result["confidence"])],
                ["异常数量", str(result["anomaly_count"])],
                ["平均强度", str(result["avg_intensity"])],
                ["信噪比", str(result["snr"])],
                ["是否通过", "✓ 是" if result["score"] >= threshold else "✗ 否"],
            ],
        })
        
        # ── Per-image result: brief document (grouped) ──
        status = "通过" if result["score"] >= threshold else "未通过"
        per_image_results.append({
            "name": "分析摘要",
            "type": "document",
            "group": name,
            "content": f"### {name} 分析结果\n\n- **状态**: {status}\n- **检测分数**: {result['score']}\n- **置信度**: {result['confidence']}\n- **异常数量**: {result['anomaly_count']}\n- **模式**: {mode}",
        })
    
    # ── Optional trend chart (no group → top-level tab) ──
    if show_trend and len(image_names) > 1:
        trend_datasets = [
            {
                "label": "检测分数",
                "data": trend_scores,
                "borderColor": PALETTE[0],
                "backgroundColor": PALETTE[0] + "33",
            },
            {
                "label": "置信度",
                "data": trend_confidences,
                "borderColor": PALETTE[1],
                "backgroundColor": PALETTE[1] + "33",
            },
        ]
        per_image_results.insert(0, {
            "name": "📈 趋势图",
            "type": "chart",
            "chartType": "line",
            "data": {
                "labels": trend_labels,
                "datasets": trend_datasets,
            },
            # No "group" → appears as top-level tab
        })
        
        # Also add a summary table (no group → top-level tab)
        summary_rows = []
        for i, name in enumerate(image_names):
            status = "通过" if trend_scores[i] >= threshold else "未通过"
            summary_rows.append([
                name, str(trend_scores[i]), str(trend_confidences[i]), status
            ])
        per_image_results.insert(1, {
            "name": "📊 汇总表",
            "type": "table",
            "columns": ["文件名", "检测分数", "置信度", "是否通过"],
            "rows": summary_rows,
            # No "group" → appears as top-level tab
        })
    
    # Use dynamic _results mode
    return {"_results": per_image_results}