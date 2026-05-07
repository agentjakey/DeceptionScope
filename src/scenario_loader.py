from __future__ import annotations

import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()

# Fields required for standard (BoN) scenarios.
_REQUIRED_STANDARD: tuple[str, ...] = (
    "name",
    "title",
    "threat_model",
    "system_prompt",
    "user_turns",
    "evaluation_dimensions",
    "bon_n",
)

# Fields required specifically for multiplayer scenarios (in place of system_prompt
# and user_turns). Base fields (name, title, etc.) are checked separately.
_REQUIRED_MULTIPLAYER: tuple[str, ...] = (
    "name",
    "title",
    "threat_model",
    "persuader_system_prompt",
    "skeptic_system_prompt",
    "opening_statement",
    "evaluation_dimensions",
    "bon_n",
)


def _is_multiplayer(scenario: dict) -> bool:
    return "persuader_system_prompt" in scenario


def _validate(scenario: dict, path: str) -> None:
    """Raise ValueError with a descriptive message if required fields are missing."""
    required = _REQUIRED_MULTIPLAYER if _is_multiplayer(scenario) else _REQUIRED_STANDARD

    missing = [field for field in required if field not in scenario or scenario[field] is None]
    if missing:
        raise ValueError(
            f"Scenario file '{path}' is missing required field(s): {', '.join(missing)}"
        )

    # evaluation_dimensions must be a non-empty list and each entry must have name + description
    dims = scenario.get("evaluation_dimensions")
    if not isinstance(dims, list) or len(dims) == 0:
        raise ValueError(
            f"Scenario file '{path}': 'evaluation_dimensions' must be a non-empty list."
        )
    for i, dim in enumerate(dims):
        if not isinstance(dim, dict) or "name" not in dim or "description" not in dim:
            raise ValueError(
                f"Scenario file '{path}': evaluation_dimensions[{i}] "
                "must have 'name' and 'description' keys."
            )

    # bon_n must be a positive integer
    bon_n = scenario.get("bon_n")
    if not isinstance(bon_n, int) or bon_n < 1:
        raise ValueError(
            f"Scenario file '{path}': 'bon_n' must be a positive integer, got {bon_n!r}."
        )

    # user_turns must be a non-empty list for standard scenarios
    if not _is_multiplayer(scenario):
        turns = scenario.get("user_turns")
        if not isinstance(turns, list) or len(turns) == 0:
            raise ValueError(
                f"Scenario file '{path}': 'user_turns' must be a non-empty list."
            )


def load_scenario(path: str) -> dict:
    """Load and validate a single YAML scenario file.

    Raises ValueError with a descriptive message if any required field is
    missing or malformed. Raises yaml.YAMLError if the file is not valid YAML.
    """
    with open(path, encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario file '{path}' did not parse to a dict.")

    _validate(scenario, path)
    return scenario


def list_scenarios(directory: str = "scenarios/") -> list[dict]:
    """Load all .yaml files in directory, print a Rich summary table, and return them.

    Files that fail validation are skipped with a warning printed to the console.
    """
    scenarios: list[dict] = []
    errors: list[str] = []

    for yaml_path in sorted(Path(directory).glob("*.yaml")):
        try:
            scenarios.append(load_scenario(str(yaml_path)))
        except (ValueError, yaml.YAMLError) as exc:
            errors.append(f"{yaml_path.name}: {exc}")

    table = Table(
        title=f"Scenarios  ({len(scenarios)} loaded)",
        show_header=True,
        header_style="dim",
        border_style="dim",
    )
    table.add_column("Name", no_wrap=True)
    table.add_column("Threat Model")
    table.add_column("N", justify="right")
    table.add_column("Tags")

    for s in scenarios:
        tags = ", ".join(s.get("tags", [])) or "-"
        scenario_type = "[dim]multiplayer[/dim]" if _is_multiplayer(s) else ""
        name_cell = s["name"] + (f"  {scenario_type}" if scenario_type else "")
        table.add_row(
            name_cell,
            s.get("threat_model", ""),
            str(s.get("bon_n", "")),
            tags,
        )

    console.print()
    console.print(table)

    for err in errors:
        console.print(f"[red]Validation error:[/red] {err}")

    console.print()
    return scenarios


def get_multiplayer_scenarios(directory: str = "scenarios/") -> list[dict]:
    """Return only scenarios that have persuader_system_prompt and skeptic_system_prompt.

    Skips files that fail validation without raising.
    """
    results: list[dict] = []
    for yaml_path in sorted(Path(directory).glob("*.yaml")):
        try:
            scenario = load_scenario(str(yaml_path))
            if _is_multiplayer(scenario):
                results.append(scenario)
        except (ValueError, yaml.YAMLError):
            continue
    return results
