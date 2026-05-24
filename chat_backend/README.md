# AlgoChat

**算法对话平台** — 上传文件，选择算法，获取结构化结果。

AlgoChat 是一个通用的算法 IO 与展示平台。它的核心思想是：**你只需要定义算法函数的输入/输出格式，平台负责文件上传、参数配置、结果渲染的全部工作。**

## ✨ 功能特性

- 🧮 **算法即插件** — 用 `@algorithm` 装饰器注册算法，自动发现、自动生成 UI
- 📁 **多文件上传** — 支持批量上传、拖拽上传、复选框选择特定文件
- 🎛️ **参数面板** — 自动根据参数定义生成滑块/输入框/下拉选择
- 📊 **多类型结果展示** — 图表（Chart）、表格（Table）、图片（Image）、文档（Document）Tab 切换
- 🔍 **预览面板** — 右侧可展开的独立预览区，支持全屏和下载
- 🌓 **浅色/深色主题** — CSS 变量驱动，一键切换
- 💬 **对话式交互** — Gemini 风格对话布局，支持关键词路由到算法
- 💾 **对话持久化** — localStorage 保存对话历史

## 🛠️ 技术栈

| 层 | 技术 |
|---|---|
| 前端 | HTML + CSS + Vanilla JS |
| 图表 | Chart.js |
| 表格导出 | SheetJS (XLSX) |
| Markdown | marked.js |
| 后端 | Python FastAPI |
| 数据处理 | pandas, scikit-learn, numpy |

## 📁 项目结构

```
demo/
├── index.html              # 主页面
├── css/
│   └── style.css           # 全局样式 + 主题系统
├── js/
│   ├── app.js              # 主应用逻辑
│   ├── api.js              # API 通信层
│   ├── particles.js        # 粒子背景效果
│   ├── charts.js           # 图表工具
│   └── table.js            # 表格工具
├── backend/
│   ├── main.py             # FastAPI 服务入口
│   ├── algochat_base.py    # 算法注册基类 + 装饰器
│   ├── requirements.txt    # Python 依赖
│   ├── test_api.py         # API 测试
│   └── algorithms/         # 算法模块目录
│       ├── __init__.py
│       ├── _utils.py       # 工具函数（假数据生成等）
│       ├── kmeans.py       # K-Means 聚类
│       ├── dbscan.py       # DBSCAN 聚类
│       ├── linear_regression.py  # 线性回归
│       ├── pca.py          # 主成分分析
│       ├── anomaly_detection.py  # 异常检测
│       └── data_stats.py   # 数据统计与可视化
└── README.md
```

## 🚀 快速开始

### 1. 启动后端

```bash
cd backend
pip install -r requirements.txt
python main.py
```

服务默认运行在 `http://localhost:8000`。

### 2. 打开前端

直接用浏览器打开 `index.html`，或使用任意静态服务器：

```bash
# 方式一：Python
python -m http.server 3000

# 方式二：Node.js
npx serve .
```

访问 `http://localhost:3000` 即可。

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/algorithms` | 获取所有已注册算法列表 |
| POST | `/api/chat` | 对话式交互（关键词路由） |
| POST | `/api/algorithm/run` | 执行指定算法 |
| GET | `/api/data/generate` | 生成模拟数据集 |
| GET | `/api/images/{filename}` | 获取图片 |
| GET | `/api/files/{file_id}/download` | 下载文件 |

### 执行算法

```
POST /api/algorithm/run
Content-Type: multipart/form-data

algorithm: "kmeans"           # 算法 ID
params: '{"n_clusters": 3}'   # JSON 字符串
files: [上传文件列表]          # 可选
```

返回格式：

```json
{
  "type": "results",
  "message": "**K-Means聚类**执行完成",
  "results": [
    {
      "id": "kmeans_chart",
      "name": "聚类散点图",
      "type": "chart",
      "chartType": "scatter",
      "data": { "labels": [...], "datasets": [...] }
    },
    {
      "id": "kmeans_table",
      "name": "聚类中心",
      "type": "table",
      "columns": ["x", "y", "cluster"],
      "rows": [[1.2, 3.4, 0], ...]
    }
  ]
}
```

## 🧩 自定义算法

在 `backend/algorithms/` 目录下新建 `.py` 文件，使用 `@algorithm` 装饰器即可自动注册：

```python
from algochat_base import algorithm, Input, Output, Param

@algorithm(
    id="my_algo",
    name="我的算法",
    category="自定义",
    icon="🚀",
    desc="这是一个示例算法",
    inputs=[
        Input(key="data", types=["csv", "xlsx"], multiple=False, desc="输入数据"),
    ],
    outputs=[
        Output(key="result_chart", type="chart", desc="结果图表"),
        Output(key="result_table", type="table", desc="结果表格"),
        Output(key="report", type="document", desc="分析报告"),
    ],
    params=[
        Param(key="threshold", type="float", default=0.5, min=0, max=1, step=0.1, label="阈值"),
        Param(key="mode", type="select", default="fast", options=["fast", "accurate"], label="模式"),
    ],
)
def my_algorithm(inputs: dict, params: dict):
    # inputs: {"data": ["/tmp/uploaded_file.csv"]}
    # params: {"threshold": 0.5, "mode": "fast"}

    # ... 你的处理逻辑 ...

    return {
        "result_chart": {
            "chartType": "bar",
            "labels": ["A", "B", "C"],
            "datasets": [{"label": "值", "data": [10, 20, 30]}]
        },
        "result_table": {
            "columns": ["类别", "数量"],
            "rows": [["A", 10], ["B", 20], ["C", 30]]
        },
        "report": "## 分析报告\n\n算法执行成功，共处理 3 个类别。"
    }
```

重启后端后，新算法会自动出现在前端列表中。

### 输出类型格式

| 类型 | 返回值格式 | 前端渲染 |
|------|-----------|---------|
| `chart` | `{"chartType": "bar\|line\|pie\|scatter", "labels": [...], "datasets": [...]}` | Chart.js 图表 |
| `table` | `{"columns": [...], "rows": [[...], ...]}` | HTML 表格（可导出 XLSX） |
| `image` | `{"src": "/api/images/xxx.png", "name": "图片名"}` | 图片展示 |
| `document` | `"markdown 文本"` 或 `{"content": "文本"}` | Markdown 渲染 |

### 参数类型

| 类型 | UI 控件 | 说明 |
|------|--------|------|
| `int` | 滑块 | 整数，支持 min/max/step |
| `float` | 滑块 | 浮点数，支持 min/max/step |
| `select` | 下拉选择 | 需提供 options 列表 |
| `text` | 文本输入 | 自由文本 |
| `bool` | 复选框 | 布尔值 |

## 🎨 界面说明

- **侧边栏**：算法列表（支持分类筛选和搜索）+ 对话历史
- **对话区**：Gemini 风格居中布局，用户消息右侧、助手消息左侧
- **算法选择器**：输入框上方内联下拉选择
- **参数面板**：选择算法后展开，可折叠
- **文件上传区**：上传后显示网格卡片，支持多选/删除/拖拽
- **Workflow Node**：算法结果以卡片形式展示在对话中，多结果用 Tab 切换
- **预览面板**：右侧可展开，支持全屏查看和下载

## ⚙️ 设置

点击侧边栏 ⚙ 按钮可配置：

- **API 地址**：后端服务地址（默认 `http://localhost:8000`）
- **粒子效果**：开/关背景粒子动画

## 📄 License

MIT