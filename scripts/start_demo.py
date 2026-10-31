from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
NPM_CLI = Path("C:/Program Files/nodejs/npm.cmd")


def start_process(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=False,
    )


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    backend_cmd = [
        str(VENV_PYTHON),
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    frontend_cmd = [
        str(NPM_CLI),
        "run",
        "dev",
        "--",
        "--host",
        "0.0.0.0",
        "--port",
        "5173",
    ]

    backend_proc = start_process(backend_cmd, ROOT, env)
    frontend_proc = start_process(frontend_cmd, FRONTEND_DIR)

    print("SIH26070 demo stack started.")
    print("Backend URL: http://localhost:8000")
    print("Frontend URL: http://localhost:5173")
    print("Press Ctrl+C to stop the demo.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping SIH26070 demo stack...")
        for proc in (backend_proc, frontend_proc):
            if proc.poll() is None:
                proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                proc.wait(timeout=10)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
