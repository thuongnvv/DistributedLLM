"""
Start API server and Streamlit UI.

Usage:
    # Start API only
    python scripts/run_all.py api

    # Start Streamlit UI only (requires API running)
    python scripts/run_all.py ui

    # Start both (API in background)
    python scripts/run_all.py all
"""
import argparse
import subprocess
import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # allow "import settings"


def start_api() -> subprocess.Popen:
    import settings
    port = settings.PORT
    print(f"[API] Starting on port {port} (LLM_MODE={settings.LLM_MODE})...")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app",
         "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(ROOT),
    )
    time.sleep(2)
    print(f"[API] PID={proc.pid}")
    return proc


def start_streamlit() -> subprocess.Popen:
    import settings
    port = int(os.getenv("STREAMLIT_PORT", "8501"))
    api_port = settings.PORT

    print(f"[UI] Starting Streamlit on port {port}...")
    print(f"[UI] API_BASE=http://localhost:{api_port}")

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(port),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false",
         ],
        cwd=str(ROOT),
        env={**os.environ, "API_BASE": f"http://localhost:{api_port}"},
    )
    print(f"[UI] PID={proc.pid}")
    print(f"[UI] Open: http://localhost:{port}")
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["api", "ui", "all"], default="all")
    args = parser.parse_args()

    api_proc = None
    ui_proc = None

    try:
        if args.target in ("api", "all"):
            api_proc = start_api()

        if args.target in ("ui", "all"):
            if args.target == "all":
                time.sleep(3)
            ui_proc = start_streamlit()

        print("\nAll services started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            if api_proc and api_proc.poll() is not None:
                print("[API] Process died!")
                break
            if ui_proc and ui_proc.poll() is not None:
                print("[UI] Process died!")
                break
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for label, proc in [("API", api_proc), ("UI", ui_proc)]:
            if proc:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"[{label}] Stopped.")


if __name__ == "__main__":
    main()
