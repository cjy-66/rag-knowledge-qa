"""
启动入口 — 支持两种模式:
  - backend: 启动 FastAPI 后端服务 (python run.py backend)
  - frontend: 启动 Streamlit 前端 (python run.py frontend)
  - all: 同时启动后端和前端 (python run.py all)
"""
import os
import sys
import subprocess
import argparse


def run_backend():
    """启动 FastAPI 后端"""
    print(">>> Starting FastAPI backend at http://localhost:8000 ...")
    print("    API docs: http://localhost:8000/docs")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ], env=env)


def run_frontend():
    """启动 Streamlit 前端"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "streamlit_app.py")
    print(">>> Starting Streamlit frontend at http://localhost:8501 ...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        frontend_path,
        "--server.port", "8501",
        "--server.address", "localhost",
    ], env=env)


def run_all():
    """同时启动前后端"""
    import threading

    print("=" * 50)
    print("  RAG Knowledge QA System")
    print("=" * 50)
    print()

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    import time
    time.sleep(2)

    print()
    run_frontend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Knowledge Base QA Launcher")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["backend", "frontend", "all"],
        default="all",
        help="Mode: backend / frontend / all (default)",
    )
    args = parser.parse_args()

    if args.mode == "backend":
        run_backend()
    elif args.mode == "frontend":
        run_frontend()
    else:
        run_all()
