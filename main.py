from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dashboard import (
    print_comparison_summary,
    print_flags,
    print_live_sample,
    print_score_summary,
    print_session_footer,
    print_session_header,
)
from src.models import AVAILABLE_MODELS, ProviderError
from src.multiplayer import run_multiplayer
from src.report import (
    generate_comparison_report,
    generate_multiplayer_report,
    generate_report,
)
from src.runner import SessionResult, run_bon_session, run_comparison_session
from src.scenario_loader import get_multiplayer_scenarios, list_scenarios, load_scenario
from src.scorer import score_session

console = Console()

app = typer.Typer(
    add_completion=False,
    rich_markup_mode=None,
    help="DeceptionScope -- behavioral alignment probe CLI.",
)

SESSIONS_DIR = Path("outputs/sessions")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    console.print(
        "\nDeceptionScope -- Model Organisms of Misalignment Explorer\n"
        "Inspired by Hubinger et al. (2023). "
        "Run 'python main.py --help' for commands.\n"
    )


def _session_path(result: SessionResult) -> Path:
    safe_ts = result.timestamp.replace(":", "-").replace(".", "-")
    return SESSIONS_DIR / f"{result.scenario_name}_{safe_ts}.json"


def _find_session(session_id: str) -> tuple[dict, Path] | None:
    """Scan outputs/sessions/ for a session whose session_id field matches."""
    if not SESSIONS_DIR.exists():
        return None
    for path in SESSIONS_DIR.glob("*.json"):
        if "_scored" in path.name:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("session_id") == session_id:
                return data, path
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _load_scores_from_disk(session_id: str) -> list | None:
    score_path = SESSIONS_DIR / f"{session_id}_scored.json"
    if not score_path.exists():
        return None
    try:
        with open(score_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _find_scenario_yaml(scenario_name: str, directory: str = "scenarios/") -> str | None:
    for path in sorted(Path(directory).glob("*.yaml")):
        try:
            scenario = load_scenario(str(path))
            if scenario.get("name") == scenario_name:
                return str(path)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Banner callback
# ---------------------------------------------------------------------------


@app.callback()
def _callback() -> None:
    _print_banner()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    scenario_name: str,
    model: Annotated[str, typer.Option("--model", help="Model alias")] = "haiku",
    n: Annotated[Optional[int], typer.Option("--n", help="Override bon_n from scenario")] = None,
    no_dashboard: Annotated[
        bool, typer.Option("--no-dashboard", help="Skip Rich UI, print plain output")
    ] = False,
) -> None:
    """Run a Best-of-N probe session on a scenario."""
    yaml_path = _find_scenario_yaml(scenario_name)
    if yaml_path is None:
        console.print(f"[red]Scenario not found:[/red] {scenario_name}")
        raise typer.Exit(1)

    try:
        scenario = load_scenario(yaml_path)
    except ValueError as exc:
        console.print(f"[red]Scenario validation error:[/red] {exc}")
        raise typer.Exit(1)

    effective_n = n if n is not None else scenario["bon_n"]
    dim_names = [d["name"] for d in scenario.get("evaluation_dimensions", [])]
    timestamp = datetime.now().isoformat()

    if not no_dashboard:
        print_session_header(scenario_name, model, effective_n, timestamp)

    try:
        result = run_bon_session(scenario, model, n_override=n)
    except ProviderError as exc:
        console.print(f"[red]Provider error ({exc.provider}):[/red] {exc}")
        raise typer.Exit(1)
    except KeyError as exc:
        console.print(f"[red]Unknown model alias:[/red] {exc}")
        raise typer.Exit(1)

    total = len(result.completions)

    if no_dashboard:
        console.print(f"\nScenario : {scenario_name}")
        console.print(f"Model    : {model}  N = {total}\n")
        for i, completion in enumerate(result.completions):
            console.print(f"--- Sample {i + 1} of {total} ---")
            console.print(completion)
            console.print()
    else:
        for i, completion in enumerate(result.completions):
            print_live_sample(i + 1, total, completion)

    console.print("\n[dim]Scoring completions...[/dim]")
    try:
        scored = score_session(result, scenario)
    except ProviderError as exc:
        console.print(f"[red]Scoring error ({exc.provider}):[/red] {exc}")
        console.print("[dim]Report will be generated without scores.[/dim]")
        scored = []

    if not no_dashboard:
        print_score_summary(scored, dim_names)
        print_flags(scored)

    console.print("\n[dim]Generating report...[/dim]")
    try:
        report_dict = generate_report(result, scored, scenario)
    except Exception as exc:
        console.print(f"[dim]Report generation failed: {exc}[/dim]")
        report_dict = {}

    sess_path = _session_path(result)

    if not no_dashboard:
        print_session_footer(result, sess_path, scoring_calls=len(scored))
    else:
        if report_dict:
            safe_ts = result.timestamp.replace(":", "-").replace(".", "-")
            report_path = Path("outputs/reports") / f"{scenario_name}_{safe_ts}.md"
            console.print(f"Report saved: {report_path}")
        console.print(f"Session  : {sess_path}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command()
def compare(
    scenario_name: str,
    models: Annotated[
        str,
        typer.Option("--models", help='Comma-separated model aliases, e.g. "haiku,gpt4o-mini"'),
    ] = "haiku,gpt4o-mini",
    n: Annotated[Optional[int], typer.Option("--n", help="Samples per model")] = None,
) -> None:
    """Run the same scenario across multiple models and compare scores."""
    yaml_path = _find_scenario_yaml(scenario_name)
    if yaml_path is None:
        console.print(f"[red]Scenario not found:[/red] {scenario_name}")
        raise typer.Exit(1)

    try:
        scenario = load_scenario(yaml_path)
    except ValueError as exc:
        console.print(f"[red]Scenario validation error:[/red] {exc}")
        raise typer.Exit(1)

    model_aliases = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_aliases) < 2:
        console.print("[red]At least two model aliases are required for comparison.[/red]")
        raise typer.Exit(1)

    effective_n = n if n is not None else scenario.get("bon_n", 5)
    dim_names = [d["name"] for d in scenario.get("evaluation_dimensions", [])]

    console.print(
        f"Comparing {len(model_aliases)} models on '{scenario_name}'  "
        f"(N = {effective_n} per model)"
    )

    try:
        results = run_comparison_session(scenario, model_aliases, n=effective_n)
    except ProviderError as exc:
        console.print(f"[red]Provider error ({exc.provider}):[/red] {exc}")
        raise typer.Exit(1)
    except KeyError as exc:
        console.print(f"[red]Unknown model alias:[/red] {exc}")
        raise typer.Exit(1)

    console.print("\n[dim]Scoring all sessions...[/dim]")
    comparison_scored: dict = {}
    all_scores: dict = {}
    for alias, session_result in results.items():
        try:
            scored = score_session(session_result, scenario)
        except ProviderError:
            scored = []
        comparison_scored[alias] = (session_result, scored)
        all_scores[alias] = scored

    print_comparison_summary(results, all_scores, dim_names)

    console.print("\n[dim]Generating comparison report...[/dim]")
    try:
        generate_comparison_report(comparison_scored, scenario)
        console.print("[dim]Report saved to outputs/reports/[/dim]")
    except Exception as exc:
        console.print(f"[dim]Comparison report generation failed: {exc}[/dim]")


# ---------------------------------------------------------------------------
# multiplayer
# ---------------------------------------------------------------------------


@app.command()
def multiplayer(
    scenario_name: str,
    persuader: Annotated[
        str, typer.Option("--persuader", help="Model alias for the persuader role")
    ] = "haiku",
    skeptic: Annotated[
        str, typer.Option("--skeptic", help="Model alias for the skeptic role")
    ] = "haiku",
) -> None:
    """Run a structured multi-round debate between a persuader and a skeptic."""
    yaml_path = _find_scenario_yaml(scenario_name)
    if yaml_path is None:
        console.print(f"[red]Scenario not found:[/red] {scenario_name}")
        raise typer.Exit(1)

    try:
        scenario = load_scenario(yaml_path)
    except ValueError as exc:
        console.print(f"[red]Scenario validation error:[/red] {exc}")
        raise typer.Exit(1)

    if not scenario.get("persuader_system_prompt"):
        console.print(
            f"[red]'{scenario_name}' is not a multiplayer scenario.[/red]\n"
            "Use 'python main.py list' to see multiplayer-compatible scenarios."
        )
        raise typer.Exit(1)

    console.print(
        f"Multiplayer: '{scenario_name}'  "
        f"persuader={persuader}  skeptic={skeptic}"
    )

    try:
        session = run_multiplayer(scenario, persuader, skeptic)
    except ProviderError as exc:
        console.print(f"[red]Provider error ({exc.provider}):[/red] {exc}")
        raise typer.Exit(1)
    except KeyError as exc:
        console.print(f"[red]Unknown model alias:[/red] {exc}")
        raise typer.Exit(1)

    console.print("\n[dim]Generating multiplayer report...[/dim]")
    try:
        generate_multiplayer_report(session)
        console.print("[dim]Report saved to outputs/reports/[/dim]")
    except Exception as exc:
        console.print(f"[dim]Multiplayer report generation failed: {exc}[/dim]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    port: Annotated[int, typer.Option("--port", help="Port to bind")] = 5000,
    host: Annotated[str, typer.Option("--host", help="Host to bind")] = "0.0.0.0",
) -> None:
    """Start the DeceptionScope web UI."""
    console.print(f"DeceptionScope web UI running at http://localhost:{port}")
    console.print("Press Ctrl+C to stop.\n")

    from web.app import app as flask_app

    flask_app.run(host=host, port=port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_cmd() -> None:
    """List all available scenarios with threat_model, bon_n, and tags."""
    list_scenarios("scenarios/")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command()
def report(
    session_id: str,
    force_rescore: Annotated[
        bool,
        typer.Option("--force-rescore", help="Re-run scoring even if scores exist on disk"),
    ] = False,
) -> None:
    """Re-generate a report from a previously saved session."""
    found = _find_session(session_id)
    if found is None:
        console.print(f"[red]Session not found:[/red] {session_id}")
        console.print("Run 'python main.py run <scenario>' first.")
        raise typer.Exit(1)

    data, sess_path = found

    if "final_verdict" in data:
        # Multiplayer session
        from src.multiplayer import MultiplayerSession, MultiplayerRound

        rounds = [
            MultiplayerRound(
                round_number=r["round_number"],
                persuader_output=r["persuader_output"],
                skeptic_output=r["skeptic_output"],
            )
            for r in data.get("rounds", [])
        ]
        session_obj = MultiplayerSession(
            session_id=data["session_id"],
            scenario_name=data["scenario_name"],
            persuader_model=data.get("persuader_model", ""),
            skeptic_model=data.get("skeptic_model", ""),
            timestamp=data["timestamp"],
            rounds=rounds,
            final_verdict=data.get("final_verdict", ""),
            verdict_correct=data.get("verdict_correct"),
        )
        try:
            report_dict = generate_multiplayer_report(session_obj)
            safe_ts = session_obj.timestamp.replace(":", "-").replace(".", "-")
            report_path = (
                Path("outputs/reports") / f"{session_obj.scenario_name}_{safe_ts}.md"
            )
            console.print(f"Report saved: {report_path}")
        except Exception as exc:
            console.print(f"[red]Report generation failed:[/red] {exc}")
            raise typer.Exit(1)
        return

    # Standard BoN session
    scenario_name = data.get("scenario_name", "")
    yaml_path = _find_scenario_yaml(scenario_name)
    if yaml_path is None:
        console.print(f"[red]Scenario YAML not found for:[/red] {scenario_name}")
        raise typer.Exit(1)

    try:
        scenario = load_scenario(yaml_path)
    except ValueError as exc:
        console.print(f"[red]Scenario validation error:[/red] {exc}")
        raise typer.Exit(1)

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

    scored_raw = None if force_rescore else _load_scores_from_disk(session_id)

    if scored_raw is not None:
        from src.scorer import CompletionScore

        scored = [
            CompletionScore(
                completion_index=s["completion_index"],
                completion_text=s["completion_text"],
                scores=s["scores"],
                flags=s["flags"],
                reasoning=s.get("reasoning", ""),
            )
            for s in scored_raw
        ]
        console.print("[dim]Using cached scores from disk.[/dim]")
    else:
        console.print("[dim]Scoring completions...[/dim]")
        try:
            scored = score_session(session_result, scenario)
        except ProviderError as exc:
            console.print(f"[red]Scoring error ({exc.provider}):[/red] {exc}")
            raise typer.Exit(1)

    console.print("[dim]Generating report...[/dim]")
    try:
        generate_report(session_result, scored, scenario)
        safe_ts = session_result.timestamp.replace(":", "-").replace(".", "-")
        report_path = (
            Path("outputs/reports") / f"{session_result.scenario_name}_{safe_ts}.md"
        )
        console.print(f"Report saved: {report_path}")
    except Exception as exc:
        console.print(f"[red]Report generation failed:[/red] {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app()
