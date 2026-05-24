"""
AlgoChat — Backend Server (FastAPI)
Auto-discovers algorithms from algorithms/ directory via @algorithm decorator
"""

from fastapi import FastAPI, UploadFile, File, Form, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import json, os, uuid, tempfile, io, httpx
from typing import Dict, List, Optional
from collections import deque

from algochat_base import discover_algorithms, ALGORITHMS
from algorithms._utils import generate_fake_data, IMAGE_DIR

app = FastAPI(title="AlgoChat Backend")

# ══════════════════════════════════════════
# LLM 配置（后端配置，前端无需传递）
# ══════════════════════════════════════════
LLM_API_URL = "https://api.deepseek.com/v1/chat/completions"  # 修改为您的API地址
LLM_API_KEY = "sk-15ddc56f5a8a4650a67277648765890d"  # 修改为您的API Key
LLM_MODEL = "deepseek-chat"  # 模型名称

# ══════════════════════════════════════════
# LLM 对话历史存储（内存中，按 conversation_id）
# ══════════════════════════════════════════
conversation_history: Dict[str, deque] = {}
MAX_HISTORY_ROUNDS = 10  # 保留最近10轮对话

def get_conversation_messages(conv_id: str) -> List[dict]:
    """获取对话历史，格式化为OpenAI消息格式"""
    if conv_id not in conversation_history:
        return []
    return list(conversation_history[conv_id])

def add_to_conversation(conv_id: str, role: str, content: str):
    """添加消息到对话历史"""
    if conv_id not in conversation_history:
        conversation_history[conv_id] = deque(maxlen=MAX_HISTORY_ROUNDS * 2)  # *2 because each round has user + assistant
    conversation_history[conv_id].append({"role": role, "content": content})

# ── CORS ──
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Static Files (Frontend) ──
# 获取项目根目录 (backend/ 的父目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS_DIR = os.path.join(PROJECT_ROOT, "css")
JS_DIR = os.path.join(PROJECT_ROOT, "js")

# 分别挂载 CSS 和 JS 目录
app.mount("/static/css", StaticFiles(directory=CSS_DIR), name="css")
app.mount("/static/js", StaticFiles(directory=JS_DIR), name="js")

# ── Auto-discover algorithms on startup ──
discover_algorithms()


# ── Frontend Route ──

@app.get("/")
async def root():
    """Serve the main frontend page."""
    index_path = os.path.join(PROJECT_ROOT, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "AlgoChat API Server", "docs": "/docs"}, 200)


# ── API Routes ──

@app.get("/api/algorithms")
async def list_algorithms():
    """Return all registered algorithms with full schema (inputs/outputs/params)."""
    return [algo.to_dict() for algo in ALGORITHMS.values()]


@app.post("/api/chat")
async def chat(request: Request):
    """
    Chat with keyword-based routing or LLM streaming.
    LLM配置从后端读取，前端只需发送 message 和 conversation_id
    """
    # 解析请求体
    content_type = request.headers.get('content-type', '')
    
    if 'application/json' in content_type:
        body = await request.json()
        message = body.get('message', '')
        conversation_id = body.get('conversation_id')
        stream = body.get('stream', True)  # 默认使用流式
    else:
        # Form data 不支持流式，返回普通响应
        form = await request.form()
        message = form.get('message', '')
        conversation_id = form.get('conversation_id')
        stream = False
    
    print(f"[CHAT] message={message[:30]}..., conv_id={conversation_id}, stream={stream}")
    
    # 检查是否是算法关键词
    msg_lower = message.lower()
    keyword_map = {
        "聚类": "kmeans", "kmeans": "kmeans",
        "dbscan": "dbscan",
        "回归": "linear_regression", "线性": "linear_regression",
        "降维": "pca", "pca": "pca",
        "异常": "anomaly_detection", "检测": "anomaly_detection",
        "统计": "data_stats", "分析": "data_stats", "图表": "data_stats", "可视化": "data_stats", "数据": "data_stats",
        "图片": "image_batch", "批处理": "image_batch", "批量": "image_batch", "图像": "image_batch",
    }
    
    for kw, algo_id in keyword_map.items():
        if kw in msg_lower:
            algo = ALGORITHMS.get(algo_id)
            if algo:
                result = algo.handler(inputs={}, params={})
                results_list = _format_results(algo, result)
                print(f"[CHAT] 触发算法: {algo_id}")
                return {"type": "results", "message": f"**{algo.name}**执行完成", "results": results_list}
    
    # 无匹配关键词，使用LLM流式对话
    print(f"[CHAT] 使用LLM对话, model={LLM_MODEL}")
    return await llm_chat_stream(message, conversation_id)


async def llm_chat_stream(message: str, conversation_id: Optional[str]):
    """Stream LLM chat response with conversation history. 使用后端配置。"""
    
    print(f"[LLM] 开始流式对话, URL={LLM_API_URL}, Model={LLM_MODEL}")
    print(f"[LLM] API Key前10位: {LLM_API_KEY[:10]}...")
    
    async def generate():
        try:
            # 准备消息历史
            messages = []
            if conversation_id:
                messages = get_conversation_messages(conversation_id)
                print(f"[LLM] 历史消息数: {len(messages)}")
            
            # 添加当前消息
            messages.append({"role": "user", "content": message})
            print(f"[LLM] 发送消息数: {len(messages)}")
            
            # 调用LLM API
            print(f"[LLM] 正在调用API...")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    LLM_API_URL,
                    headers={
                        "Authorization": f"Bearer {LLM_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": LLM_MODEL,
                        "messages": messages,
                        "stream": True
                    },
                    timeout=30.0
                )
                
                print(f"[LLM] API响应状态: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"[LLM] API错误: {error_text[:200]}")
                    yield f"data: {json.dumps({'error': f'LLM API错误 {response.status_code}: {error_text[:100]}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                print(f"[LLM] 开始接收流式响应...")
                assistant_content = ""
                chunk_count = 0
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            print(f"[LLM] 流结束, 共{chunk_count}块, 内容长度{len(assistant_content)}")
                            # 保存对话历史
                            if conversation_id:
                                add_to_conversation(conversation_id, "user", message)
                                add_to_conversation(conversation_id, "assistant", assistant_content)
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            chunk = json.loads(data)
                            if chunk.get("choices") and chunk["choices"][0].get("delta"):
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta:
                                    assistant_content += delta["content"]
                                    chunk_count += 1
                                    if chunk_count <= 3 or chunk_count % 10 == 0:
                                        print(f"[LLM] 收到块 #{chunk_count}: {delta['content'][:20]}...")
                                    yield f"data: {json.dumps(chunk)}\n\n"
                        except json.JSONDecodeError:
                            continue
                        
        except Exception as e:
            print(f"[LLM] 异常: {str(e)}")
            yield f"data: {json.dumps({'error': f'请求失败: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/algorithm/run")
async def run_algorithm(algorithm: str = Form(...), params: str = Form("{}"), files: list[UploadFile] = File(default=[])):
    """Run a specific algorithm with uploaded files and parameters."""
    algo = ALGORITHMS.get(algorithm)
    if not algo:
        return {"type": "error", "message": f"未知算法: {algorithm}"}

    parsed_params = json.loads(params)

    # Save uploaded files to temp paths, grouped by input key
    saved_files = []
    for f in files:
        path = os.path.join(tempfile.gettempdir(), f"algochat_{uuid.uuid4().hex[:8]}_{f.filename}")
        with open(path, "wb") as out:
            out.write(await f.read())
        saved_files.append(path)

    # Build inputs dict — map files to the first input key
    inputs = {}
    if algo.inputs:
        first_input = algo.inputs[0]
        if first_input.multiple:
            inputs[first_input.key] = saved_files
        else:
            inputs[first_input.key] = saved_files[:1]  # single file as list
    else:
        inputs["data"] = saved_files

    try:
        result = algo.handler(inputs=inputs, params=parsed_params)
        results_list = _format_results(algo, result)
        return {"type": "results", "message": f"**{algo.name}**执行完成", "results": results_list}
    except Exception as e:
        return {"type": "error", "message": str(e)}


def _format_results(algo_def, result: dict) -> list:
    """Format algorithm return dict into the API results list.

    The algorithm handler returns {output_key: value}.
    We match each key to the declared Output to determine type and name.
    
    If the handler returns a list of dicts instead, each dict must contain
    'key' + type-specific fields + optional 'group'. This enables dynamic
    results whose count is determined at runtime (e.g. per-input-file).
    """
    outputs = []

    # ── Dynamic results mode ──
    # If result contains a special "_results" key with a list, use that directly.
    # This allows algorithms to return a variable number of results at runtime.
    dynamic = result.get("_results")
    if isinstance(dynamic, list):
        for i, item in enumerate(dynamic):
            out = {
                "id": f"{algo_def.id}_dyn_{i}",
                "name": item.get("name", f"结果 {i+1}"),
                "type": item.get("type", "document"),
            }
            # Copy group if present
            if "group" in item:
                out["group"] = item["group"]
            # Copy type-specific fields
            for k in ("chartType", "data", "src", "columns", "rows", "content"):
                if k in item:
                    out[k] = item[k]
            outputs.append(out)
        return outputs

    # ── Static results mode (original) ──
    for out_def in algo_def.outputs:
        value = result.get(out_def.key)
        if value is None:
            continue

        item = {"id": f"{algo_def.id}_{out_def.key}", "name": out_def.desc, "type": out_def.type}
        # Pass group from Output definition
        if out_def.group:
            item["group"] = out_def.group

        if out_def.type == "chart":
            if isinstance(value, dict):
                item["chartType"] = value.get("chartType", "bar")
                if "datasets" in value:
                    item["data"] = {"datasets": value["datasets"]}
                    if "labels" in value:
                        item["data"]["labels"] = value["labels"]
                # Allow dynamic group from value
                if "group" in value:
                    item["group"] = value["group"]
        elif out_def.type == "image":
            if isinstance(value, dict):
                item["src"] = value.get("src", "")
                item["name"] = value.get("name", out_def.desc)
                if "group" in value:
                    item["group"] = value["group"]
        elif out_def.type == "table":
            if isinstance(value, dict):
                item["columns"] = value.get("columns", [])
                item["rows"] = value.get("rows", [])
                if "group" in value:
                    item["group"] = value["group"]
        elif out_def.type == "document":
            if isinstance(value, str):
                item["content"] = value
            elif isinstance(value, dict):
                item["content"] = value.get("content", str(value))
                if "group" in value:
                    item["group"] = value["group"]

        outputs.append(item)
    return outputs


@app.get("/api/data/generate")
async def generate_data(dataset: str = Query("clusters"), n_samples: int = Query(200)):
    try:
        df = generate_fake_data(dataset, n_samples)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 400)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(io.BytesIO(output.getvalue().encode("utf-8-sig")), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=fake_{dataset}_{n_samples}.csv"})


@app.get("/api/images/{filename}")
async def get_image(filename: str):
    path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "图片未找到"}, 404)


@app.get("/api/files/{file_id}/download")
async def download_file(file_id: str):
    path = os.path.join(tempfile.gettempdir(), file_id)
    if os.path.exists(path):
        return FileResponse(path, filename=os.path.basename(path))
    return JSONResponse({"error": "文件未找到"}, 404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)