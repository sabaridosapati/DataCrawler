"""
stop.py - Stop all Data Crawler services.
Usage: python stop.py
"""

import subprocess
import sys

# Window titles match what was set in start.py
SERVICES = [
    {"name": "Orchestrator API",      "port": 8000},
    {"name": "Docling Service",        "port": 8004},
    {"name": "Embedding Service",      "port": 8002},
    {"name": "LLM Service",            "port": 8001},
    {"name": "Knowledge Graph Service","port": 8003},
    {"name": "Frontend",               "port": 8501},
]


def kill_by_window_title(name):
    """Kill the cmd window and its child processes by window title."""
    result = subprocess.run(
        f'taskkill /fi "WindowTitle eq {name}*" /t /f',
        shell=True,
        capture_output=True,
        text=True
    )
    if "SUCCESS" in result.stdout:
        print(f"  [STOPPED] {name}")
    else:
        print(f"  [NOT FOUND] {name} (may already be stopped)")


def kill_by_port(port):
    """Kill any process still listening on a port (fallback)."""
    result = subprocess.run(
        f'netstat -ano | findstr :{port}',
        shell=True,
        capture_output=True,
        text=True
    )
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if parts and parts[-1].isdigit():
            pid = parts[-1]
            subprocess.run(f'taskkill /PID {pid} /f', shell=True,
                           capture_output=True)


def main():
    print("=" * 50)
    print("  Data Crawler - Stopping All Services")
    print("=" * 50)
    print()

    for svc in SERVICES:
        kill_by_window_title(svc["name"])
        kill_by_port(svc["port"])

    print()
    print("All services stopped.")
    print()
    print("NOTE: Milvus (Docker) and Neo4j Desktop are NOT")
    print("stopped automatically. Stop them manually if needed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
