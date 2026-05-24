#!/usr/bin/env python3
"""
AlgoChat - 一键启动服务器
运行: python server.py
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

PORT = 8001  # 使用 8001 端口避免冲突

def check_python():
    """检查 Python 版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[错误] 需要 Python 3.8+，当前版本: {}.{}.{}".format(version.major, version.minor, version.micro))
        sys.exit(1)
    print(f"[✓] Python {version.major}.{version.minor}.{version.micro}")


def install_dependencies():
    """安装依赖"""
    req_file = Path(__file__).parent / "backend" / "requirements.txt"
    if not req_file.exists():
        print("[警告] 未找到 requirements.txt")
        return
    
    print("[*] 检查依赖...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            check=True,
            capture_output=True
        )
        print("[✓] 依赖已就绪")
    except subprocess.CalledProcessError:
        print("[!] 依赖安装可能有警告，继续启动...")


def start_server():
    """启动服务器"""
    backend_dir = Path(__file__).parent / "backend"
    main_file = backend_dir / "main.py"
    
    if not main_file.exists():
        print(f"[错误] 未找到 {main_file}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("   AlgoChat 服务器已启动")
    print("=" * 50)
    print(f"  访问地址: http://localhost:{PORT}")
    print(f"  API文档:  http://localhost:{PORT}/docs")
    print("=" * 50)
    print("  按 Ctrl+C 停止服务器\n")
    
    # 延迟打开浏览器
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{PORT}")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动 uvicorn (使用 subprocess 避免模块导入问题)
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0", "--port", str(PORT), "--app-dir", str(backend_dir)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[错误] 服务器启动失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("[错误] 未找到 uvicorn，请运行: pip install uvicorn")
        sys.exit(1)


def main():
    print("=" * 50)
    print("   AlgoChat - 一键启动")
    print("=" * 50)
    
    check_python()
    install_dependencies()
    start_server()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[✓] 服务器已停止")
        sys.exit(0)