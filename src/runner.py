from __future__ import annotations

import json
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

# Ensure project root is on sys.path so this module works when run directly
# via `python src/runner.py` as well as when imported as `src.runner`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import AVAILABLE_MODELS, complete  # noqa: E402

console = Console()

SESSIONS_DIR = Path("outputs/sessions")


@dataclass
class SessionResult:
    session_id: str
    scenario_name: str
    model_alias: str
    timestamp: str
    system_prompt: str
    user_turns: list[str]
    completions: list[str]
    metadata: dict


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _safe_timestamp(ts: str) -> str:
    """Convert ISO timestamp to a filesystem-safe string."""
    return ts.replace(":", "-").replace(".", "-")


def run_bon_session(
    scenario: dict,
    model_alias: str,
    n_override: int | None = None,
) -> SessionResult:
    """Run Best-of-N sampling for a scenario and return the aggregated result.

    Calls complete() n times sequentially, displays a progress bar, and saves
    the session to outputs/sessions/{scenario_name}_{timestamp}.json.
    Raises ProviderError if any individual call fails.
    """
    n = n_override if n_override is not None else scenario["bon_n"]
    scenario_name: str = scenario["name"]
    system_prompt: str = scenario["system_prompt"]
    user_turns: list[str] = scenario["user_turns"]
    config = AVAILABLE_MODELS[model_alias]

    session_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    completions: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            f"Sampling 0 / {n}  [{model_alias}]",
            total=n,
        )
        for i in range(n):
            progress.update(
                task_id,
                description=f"Sampling {i + 1} / {n}  [{model_alias}]",
            )
            completions.append(complete(system_prompt, user_turns, model_alias))
            progress.advance(task_id)

    total_output_chars = sum(len(c) for c in completions)
    estimated_output_tokens = int(total_output_chars / 4)
    estimated_cost = round(
        (estimated_output_tokens / 1000) * config.cost_per_1k_tokens, 6
    )

    result = SessionResult(
        session_id=session_id,
        scenario_name=scenario_name,
        model_alias=model_alias,
        timestamp=timestamp,
        system_prompt=system_prompt,
        user_turns=user_turns,
        completions=completions,
        metadata={
            "n_samples": n,
            "model_id": config.model_id,
            "provider": config.provider.value,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": estimated_cost,
        },
    )

    output_path = SESSIONS_DIR / f"{scenario_name}_{_safe_timestamp(timestamp)}.json"
    _save_json(asdict(result), output_path)
    console.print(f"Session saved: [dim]{output_path}[/dim]")

    return result


def run_comparison_session(
    scenario: dict,
    model_aliases: list[str],
    n: int = 5,
) -> dict[str, SessionResult]:
    """Run the same scenario against multiple model aliases.

    Saves each individual SessionResult to its own file, then writes a combined
    summary to outputs/sessions/comparison_{scenario_name}_{timestamp}.json.
    Returns a dict keyed by model alias.
    """
    scenario_name: str = scenario["name"]
    timestamp = datetime.now().isoformat()
    results: dict[str, SessionResult] = {}

    for alias in model_aliases:
        console.print(f"\n[bold]{scenario_name}[/bold]  ->  [cyan]{alias}[/cyan]")
        results[alias] = run_bon_session(scenario, alias, n_override=n)

    combined = {
        "scenario_name": scenario_name,
        "timestamp": timestamp,
        "model_aliases": model_aliases,
        "n_per_model": n,
        "results": {alias: asdict(result) for alias, result in results.items()},
    }

    combined_path = (
        SESSIONS_DIR / f"comparison_{scenario_name}_{_safe_timestamp(timestamp)}.json"
    )
    _save_json(combined, combined_path)
    console.print(f"\nComparison session saved: [dim]{combined_path}[/dim]")

    return results


if __name__ == "__main__":
    import yaml

    scenario_path = Path("scenarios/sycophancy_probe.yaml")
    with open(scenario_path, encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    console.print("[bold]Smoke test: sycophancy_probe on haiku, n=3[/bold]")
    result = run_bon_session(scenario, model_alias="haiku", n_override=3)
    console.print(f"[green]Done.[/green] session_id={result.session_id}")
    console.print(f"Completions collected: {len(result.completions)}")
