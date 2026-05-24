@echo off
chcp 65001 >nul
title AlgoChat Server
echo ========================================
echo    AlgoChat - 一键启动脚本
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [1/3] 检测到 Python:
python --version
echo.

:: Install dependencies
echo [2/3] 安装依赖...
cd backend
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [警告] 依赖安装可能有问题，继续尝试启动...
)
cd ..
echo.

:: Start server
echo [3/3] 启动服务器...
echo.
echo 服务端点: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务器
echo ========================================
echo.

cd backend
python main.py