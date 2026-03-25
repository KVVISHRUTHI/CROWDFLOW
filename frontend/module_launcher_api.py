import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, request

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    name: str
    script: str
    description: str


MODULE_SPECS: Dict[str, ModuleSpec] = {
    "live": ModuleSpec(
        key="live",
        name="Live Detection + Prediction",
        script="main.py",
        description="Runs live crowd detection and immediate prediction module.",
    ),
    "timeline": ModuleSpec(
        key="timeline",
        name="Timeline Studio Prediction",
        script="main2.py",
        description="Runs timeline-driven prediction studio module.",
    ),
    "geospatial": ModuleSpec(
        key="geospatial",
        name="Geospatial Dashboard",
        script="geospatial_dashboard.py",
        description="Runs geospatial intelligence dashboard module.",
    ),
}


app = Flask(__name__)

# Tracks active process per module key.
module_processes: Dict[str, subprocess.Popen] = {}


def _is_process_running(process: Optional[subprocess.Popen]) -> bool:
    return bool(process is not None and process.poll() is None)


def _cleanup_finished_processes() -> None:
    finished = [key for key, proc in module_processes.items() if not _is_process_running(proc)]
    for key in finished:
        module_processes.pop(key, None)


def _module_state(key: str) -> Dict[str, object]:
    spec = MODULE_SPECS[key]
    proc = module_processes.get(key)
    running = _is_process_running(proc)
    return {
        "key": spec.key,
        "name": spec.name,
        "script": spec.script,
        "description": spec.description,
        "status": "running" if running else "stopped",
        "pid": proc.pid if running and proc is not None else None,
    }


def _build_python_command(script_name: str) -> list[str]:
    script_path = ROOT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    return [sys.executable, str(script_path)]


def _start_module(key: str) -> Dict[str, object]:
    _cleanup_finished_processes()
    current = module_processes.get(key)
    if _is_process_running(current):
        return _module_state(key)

    spec = MODULE_SPECS[key]
    command = _build_python_command(spec.script)
    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        env=os.environ.copy(),
    )
    module_processes[key] = process
    return _module_state(key)


def _stop_module(key: str) -> Dict[str, object]:
    _cleanup_finished_processes()
    process = module_processes.get(key)
    if not _is_process_running(process):
        module_processes.pop(key, None)
        return _module_state(key)

    assert process is not None
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

    module_processes.pop(key, None)
    return _module_state(key)


@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", methods=["GET"])
def root_index():
    _cleanup_finished_processes()
    return jsonify(
        {
            "service": "Crowd System AI Module Launcher",
            "status": "ok",
            "project": "Crowd_System_AI",
            "endpoints": {
                "health": "/api/health",
                "modules": "/api/modules",
                "start": "/api/modules/<module_key>/start",
                "stop": "/api/modules/<module_key>/stop",
                "stop_all": "/api/modules/stop-all",
            },
            "module_keys": list(MODULE_SPECS.keys()),
            "active_modules": [_module_state(key) for key in MODULE_SPECS.keys()],
        }
    )


@app.route("/favicon.ico", methods=["GET"])
def favicon_no_content():
    return ("", 204)


@app.route("/api/health", methods=["GET", "OPTIONS"])
def health_check():
    if request.method == "OPTIONS":
        return ("", 204)
    _cleanup_finished_processes()
    return jsonify({"status": "ok", "project": "Crowd_System_AI"})


@app.route("/api/modules", methods=["GET", "OPTIONS"])
def list_modules():
    if request.method == "OPTIONS":
        return ("", 204)
    _cleanup_finished_processes()
    return jsonify({"modules": [_module_state(key) for key in MODULE_SPECS.keys()]})


@app.route("/api/modules/<module_key>/start", methods=["POST", "OPTIONS"])
def start_module(module_key: str):
    if request.method == "OPTIONS":
        return ("", 204)
    if module_key not in MODULE_SPECS:
        return jsonify({"error": f"Unknown module: {module_key}"}), 404
    try:
        state = _start_module(module_key)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(state)


@app.route("/api/modules/<module_key>/stop", methods=["POST", "OPTIONS"])
def stop_module(module_key: str):
    if request.method == "OPTIONS":
        return ("", 204)
    if module_key not in MODULE_SPECS:
        return jsonify({"error": f"Unknown module: {module_key}"}), 404
    state = _stop_module(module_key)
    return jsonify(state)


@app.route("/api/modules/stop-all", methods=["POST", "OPTIONS"])
def stop_all_modules():
    if request.method == "OPTIONS":
        return ("", 204)
    _cleanup_finished_processes()
    for key in list(module_processes.keys()):
        _stop_module(key)
    return jsonify({"status": "stopped", "modules": [_module_state(key) for key in MODULE_SPECS.keys()]})


if __name__ == "__main__":
    host = os.environ.get("CPS_API_HOST", "127.0.0.1")
    port = int(os.environ.get("CPS_API_PORT", "5001"))
    app.run(host=host, port=port, debug=False)
