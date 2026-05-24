#!/bin/bash

# AlgoChat - 一键启动脚本 (Linux/Mac)

set -e

echo "========================================"
echo "   AlgoChat - 一键启动脚本"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3，请先安装 Python 3.8+"
    exit 1
fi

echo "[1/3] 检测到 Python:"
python3 --version
echo ""

# Install dependencies
echo "[2/3] 安装依赖..."
cd "$(dirname "$0")/backend"
pip3 install -r requirements.txt -q || {
    echo "[警告] 依赖安装可能有问题，继续尝试启动..."
}
cd ..
echo ""

# Start server
echo "[3/3] 启动服务器..."
echo ""
echo "服务端点: http://localhost:8000"
echo "API文档:  http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "========================================"
echo ""

cd backend
python3 main.py