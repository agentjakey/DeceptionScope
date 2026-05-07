from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

# Add project root to sys.path before importing src modules.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")
from src.models import ProviderError, list_available_models  # noqa: E402
from src.multiplayer import run_multiplayer  # noqa: E402
from src.runner import SessionResult, run_bon_session, run_comparison_session  # noqa: E402
from src.scorer import score_session  # noqa: E402

_HERE = Path(__file__).resolve().parent
SCENARIOS_DIR = _ROOT / "scenarios"
SESSIONS_DIR = _ROOT / "outputs" / "sessions"
REPORTS_DIR = _ROOT / "outputs" / "reports"

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)
CORS(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri="memory://",
    strategy="fixed-window",
)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded. Please wait before running another session."}), 429

# When FORWARDED_ALLOW_IPS=* is set (e.g. Railway), trust X-Forwarded-For so
# get_remote_address returns the real client IP rather than the proxy's address.
if os.environ.get("FORWARDED_ALLOW_IPS") == "*":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_all_scenarios() -> list[dict]:
    scenarios: list[dict] = []
    for path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        try:
            with open(path, encoding="utf-8") as f:
                scenarios.append(yaml.safe_load(f))
        except yaml.YAMLError:
            continue
    return scenarios


def _load_scenario(scenario_name: str) -> dict | None:
    for path in SCENARIOS_DIR.glob("*.yaml"):
        try:
            with open(path, encoding="utf-8") as f:
                scenario = yaml.safe_load(f)
            if scenario.get("name") == scenario_name:
                return scenario
        except yaml.YAMLError:
            continue
    return None


def _find_session_data(session_id: str) -> dict | None:
    """Scan outputs/sessions/ for a file whose session_id field matches."""
    if not SESSIONS_DIR.exists():
        return None
    for path in SESSIONS_DIR.glob("*.json"):
        if "_scored" in path.name:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("session_id") == session_id:
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _load_scores(session_id: str) -> list[dict] | None:
    score_path = SESSIONS_DIR / f"{session_id}_scored.json"
    if not score_path.exists():
        return None
    try:
        with open(score_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _session_metadata(data: dict) -> dict:
    """Extract lightweight metadata from a raw session dict."""
    if "final_verdict" in data:
        session_type = "multiplayer"
        count = len(data.get("rounds", []))
        model = data.get("persuader_model", "")
    elif "results" in data:
        session_type = "comparison"
        count = sum(
            len(r.get("completions", []))
            for r in data.get("results", {}).values()
        )
        model = ", ".join(data.get("model_aliases", []))
    else:
        session_type = "bon"
        count = len(data.get("completions", []))
        model = data.get("model_alias", "")

    return {
        "session_id": data.get("session_id", ""),
        "scenario_name": data.get("scenario_name", ""),
        "model": model,
        "timestamp": data.get("timestamp", ""),
        "completion_count": count,
        "session_type": session_type,
    }


# ---------------------------------------------------------------------------
# Static routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    template_path = _HERE / "templates" / "index.html"
    if not template_path.exists():
        return (
            "<h1>DeceptionScope</h1>"
            "<p>Frontend not yet built. Run the CLI with "
            "<code>python main.py run &lt;scenario&gt;</code>.</p>"
        ), 200
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Read-only API
# ---------------------------------------------------------------------------


@app.route("/api/scenarios")
def get_scenarios():
    return jsonify(_load_all_scenarios())


@app.route("/api/models")
def get_models():
    return jsonify(list_available_models())


@app.route("/api/sessions")
def get_sessions():
    if not SESSIONS_DIR.exists():
        return jsonify([])

    sessions: list[dict] = []
    paths = sorted(
        SESSIONS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        if "_scored" in path.name:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            sessions.append(_session_metadata(data))
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    return jsonify(sessions)


@app.route("/api/sessions/<session_id>")
def get_session(session_id: str):
    data = _find_session_data(session_id)
    if data is None:
        return jsonify({"error": "Session not found"}), 404

    scores = _load_scores(session_id)
    if scores is not None:
        data["scores"] = scores

    return jsonify(data)


@app.route("/api/report/<session_id>")
def get_report(session_id: str):
    data = _find_session_data(session_id)
    if data is None:
        return jsonify({"error": "Session not found"}), 404

    if not REPORTS_DIR.exists():
        return jsonify({"error": "Report not found"}), 404

    scenario_name = data.get("scenario_name", "")
    candidates = sorted(
        REPORTS_DIR.glob(f"{scenario_name}_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            return path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/plain"}
        except OSError:
            continue

    return jsonify({"error": "Report not found"}), 404


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------


@app.route("/api/run", methods=["POST"])
@limiter.limit("10 per minute")
def run_session():
    body: dict = request.get_json(force=True) or {}
    scenario_name: str = body.get("scenario_name", "")
    model_alias: str = body.get("model_alias", "")
    n: int = int(body.get("n", 5))

    scenario = _load_scenario(scenario_name)
    if scenario is None:
        return jsonify({"error": f"Scenario not found: {scenario_name}"}), 404

    try:
        result = run_bon_session(scenario, model_alias, n_override=n)
        return jsonify(asdict(result))
    except ProviderError as exc:
        return jsonify({"error": "Provider key not configured", "provider": exc.provider}), 400
    except KeyError as exc:
        return jsonify({"error": f"Unknown model alias: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": f"Run failed: {exc}"}), 500


@app.route("/api/run-comparison", methods=["POST"])
@limiter.limit("5 per minute")
def run_comparison():
    body: dict = request.get_json(force=True) or {}
    scenario_name: str = body.get("scenario_name", "")
    model_aliases: list[str] = body.get("model_aliases", [])
    n: int = int(body.get("n", 5))

    scenario = _load_scenario(scenario_name)
    if scenario is None:
        return jsonify({"error": f"Scenario not found: {scenario_name}"}), 404

    try:
        results = run_comparison_session(scenario, model_aliases, n=n)
        return jsonify({alias: asdict(r) for alias, r in results.items()})
    except ProviderError as exc:
        return jsonify({"error": "Provider key not configured", "provider": exc.provider}), 400
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/run-multiplayer", methods=["POST"])
@limiter.limit("5 per minute")
def run_multiplayer_session():
    body: dict = request.get_json(force=True) or {}
    scenario_name: str = body.get("scenario_name", "")
    persuader_alias: str = body.get("persuader_alias", "")
    skeptic_alias: str = body.get("skeptic_alias", "")

    scenario = _load_scenario(scenario_name)
    if scenario is None:
        return jsonify({"error": f"Scenario not found: {scenario_name}"}), 404

    try:
        session = run_multiplayer(scenario, persuader_alias, skeptic_alias)
        return jsonify(asdict(session))
    except ProviderError as exc:
        return jsonify({"error": "Provider key not configured", "provider": exc.provider}), 400
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/score", methods=["POST"])
@limiter.limit("10 per minute")
def score_session_endpoint():
    body: dict = request.get_json(force=True) or {}
    session_id: str = body.get("session_id", "")

    data = _find_session_data(session_id)
    if data is None:
        return jsonify({"error": "Session not found"}), 404

    if "final_verdict" in data:
        return jsonify({"error": "Multiplayer sessions cannot be scored via this endpoint"}), 400

    if "results" in data:
        return jsonify({"error": "Comparison sessions cannot be scored via this endpoint"}), 400

    scenario_name = data.get("scenario_name", "")
    scenario = _load_scenario(scenario_name)
    if scenario is None:
        return jsonify({"error": f"Scenario not found: {scenario_name}"}), 404

    try:
        session_result = SessionResult(
            session_id=data["session_id"],
            scenario_name=data["scenario_name"],
            model_alias=data.get("model_alias", ""),
            timestamp=data["timestamp"],
            system_prompt=data["system_prompt"],
            user_turns=data["user_turns"],
            completions=data["completions"],
            metadata=data.get("metadata", {}),
        )
        scores = score_session(session_result, scenario)
        return jsonify([asdict(s) for s in scores])
    except ProviderError as exc:
        return jsonify({"error": "Provider key not configured", "provider": exc.provider}), 400
    except (KeyError, TypeError) as exc:
        return jsonify({"error": f"Session data malformed: {exc}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
