"""
start.py - Start all Data Crawler services in separate terminal windows.
Usage: python start.py
"""

import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(BASE_DIR, "datacrawlerenv", "Scripts", "python.exe")

# Verify virtual environment exists
if not os.path.exists(PYTHON):
    print(f"[ERROR] Virtual environment not found at: {PYTHON}")
    print("Make sure 'datacrawlerenv' exists in the project root.")
    sys.exit(1)

SERVICES = [
    {
        "name": "Orchestrator API",
        "cwd": os.path.join(BASE_DIR, "orchestrator_api"),
        "args": ["-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
        "delay": 2,
    },
    {
        "name": "Docling Service",
        "cwd": os.path.join(BASE_DIR, "gpu_services", "docling_service"),
        "args": ["-m", "uvicorn", "main:app", "--reload", "--port", "8004"],
        "delay": 2,
    },
    {
        "name": "Embedding Service",
        "cwd": os.path.join(BASE_DIR, "gpu_services", "embedding_service"),
        "args": ["-m", "uvicorn", "main:app", "--reload", "--port", "8002"],
        "delay": 1,
    },
    {
        "name": "LLM Service",
        "cwd": os.path.join(BASE_DIR, "gpu_services", "llm_service"),
        "args": ["-m", "uvicorn", "main:app", "--reload", "--port", "8001"],
        "delay": 1,
    },
    {
        "name": "Knowledge Graph Service",
        "cwd": os.path.join(BASE_DIR, "gpu_services", "knowledge_graph_service"),
        "args": ["-m", "uvicorn", "main:app", "--reload", "--port", "8003"],
        "delay": 1,
    },
    {
        "name": "Frontend",
        "cwd": os.path.join(BASE_DIR, "frontend"),
        "args": ["-m", "streamlit", "run", "app.py"],
        "delay": 0,
    },
]


def start_service(name, cwd, args):
    cmd = " ".join([f'"{PYTHON}"'] + args)
    # Opens each service in a new named cmd window — /k keeps it open on error
    full_cmd = f'start "{name}" cmd /k "{cmd}"'
    subprocess.Popen(full_cmd, shell=True, cwd=cwd)
    print(f"  [OK] {name} started")


def main():
    print("=" * 50)
    print("  Data Crawler - Starting All Services")
    print("=" * 50)
    print()
    print("NOTE: Make sure the following are already running:")
    print("  - Docker Desktop  (Milvus)")
    print("  - Neo4j Desktop   (Graph DB)")
    print()

    for svc in SERVICES:
        start_service(svc["name"], svc["cwd"], svc["args"])
        if svc["delay"] > 0:
            time.sleep(svc["delay"])

    print()
    print("=" * 50)
    print("  All services launched in separate windows.")
    print()
    print("  URLs:")
    print("    Orchestrator API  ->  http://localhost:8000")
    print("    API Docs          ->  http://localhost:8000/docs")
    print("    Frontend UI       ->  http://localhost:8501")
    print("    Docling           ->  http://localhost:8004")
    print("    Embedding         ->  http://localhost:8002")
    print("    LLM               ->  http://localhost:8001")
    print("    Knowledge Graph   ->  http://localhost:8003")
    print()
    print("  Run 'python stop.py' to stop all services.")
    print("=" * 50)


if __name__ == "__main__":
    main()
