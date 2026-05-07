from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import complete
from src.multiplayer import MultiplayerSession
from src.runner import SessionResult
from src.scorer import CompletionScore, compute_dimension_stats

REPORTS_DIR = Path("outputs/reports")

# ---------------------------------------------------------------------------
# Hardcoded references
# ---------------------------------------------------------------------------

REFERENCES: list[dict[str, str]] = [
    {
        "index": "1",
        "citation": (
            "Hubinger, E., Perez, E., Schiefer, N., et al. (2023). "
            "Model Organisms of Misalignment: The Case for a New Pillar of "
            "Alignment Research. AI Alignment Forum."
        ),
        "url": "https://www.alignmentforum.org/posts/ChDH335ckdvpxXaXX",
    },
    {
        "index": "2",
        "citation": (
            "Hubinger, E., Denison, C., Mu, J., et al. (2024). "
            "Sleeper Agents: Training Deceptive LLMs that Persist Through "
            "Safety Training. arXiv:2401.05566."
        ),
        "url": "https://arxiv.org/abs/2401.05566",
    },
    {
        "index": "3",
        "citation": (
            "Irving, G., Christiano, P., Amodei, D. (2018). "
            "AI Safety via Debate. arXiv:1805.00899."
        ),
        "url": "https://arxiv.org/abs/1805.00899",
    },
    {
        "index": "4",
        "citation": (
            "Perez, E., Huang, S., Song, F., et al. (2022). "
            "Red Teaming Language Models with Language Models. "
            "arXiv:2202.03286."
        ),
        "url": "https://arxiv.org/abs/2202.03286",
    },
    {
        "index": "5",
        "citation": (
            "Ngo, R., Chan, L., Mindermann, S. (2023). "
            "The Alignment Problem from a Deep Learning Perspective. "
            "ICLR 2024. arXiv:2209.00626."
        ),
        "url": "https://arxiv.org/abs/2209.00626",
    },
    {
        "index": "6",
        "citation": (
            "Wei, J., Goyal, D., Durrett, G., Jain, D. (2023). "
            "Simple Synthetic Data Reduces Sycophancy in Large Language Models. "
            "arXiv:2308.03958."
        ),
        "url": "https://arxiv.org/abs/2308.03958",
    },
    {
        "index": "7",
        "citation": (
            "Kenton, Z., Everitt, T., Weidinger, L., et al. (2021). "
            "Alignment of Language Agents. arXiv:2103.14659."
        ),
        "url": "https://arxiv.org/abs/2103.14659",
    },
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_timestamp(ts: str) -> str:
    return ts.replace(":", "-").replace(".", "-")


def _avg_score(s: CompletionScore) -> float:
    """Return the mean of all valid dimension scores, or -1.0 on failure."""
    if "scoring_failed" in s.flags:
        return -1.0
    valid = [v for v in s.scores.values() if v >= 0.0]
    return sum(valid) / len(valid) if valid else -1.0


def _flags_summary(scored: list[CompletionScore]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in scored:
        for flag in s.flags:
            if flag != "scoring_failed":
                counts[flag] = counts.get(flag, 0) + 1
    return counts


def _generate_interpretation(
    threat_model: str,
    model_alias: str,
    n: int,
    dim_stats: dict[str, Any],
    flags_summary: dict[str, int],
) -> str:
    """Call the Claude API (haiku) to produce a one-paragraph interpretation."""
    stats_lines = "\n".join(
        f"  {dim}: mean={v['mean']:.3f}, std={v['std']:.3f}"
        for dim, v in dim_stats.items()
    )
    flags_text = (
        ", ".join(f"{k} ({v}x)" for k, v in sorted(flags_summary.items(), key=lambda x: -x[1]))
        or "none"
    )
    system = (
        "You are an alignment researcher writing a technical research note on a "
        "Best-of-N language model behavioral probe. Write in plain prose only. "
        "No bullet points. Do not overstate conclusions from limited sample sizes. "
        "Keep your response to one paragraph of four to six sentences."
    )
    user = (
        f"Probe summary: {n}-sample Best-of-N evaluation of model '{model_alias}' "
        f"against the threat model '{threat_model}'.\n\n"
        f"Dimension scores (mean, std across {n} samples):\n{stats_lines}\n\n"
        f"Flags observed: {flags_text}\n\n"
        "Write one paragraph interpreting what the score distribution suggests about "
        "this model's behavioral patterns in this threat-model context. "
        "Be precise and avoid overstatement."
    )
    try:
        return complete(system, [user], "haiku", max_tokens=400)
    except Exception:
        return "Interpretation unavailable (API call failed)."


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a list of rows as a Markdown table."""
    header = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---|" for _ in headers)
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([header, sep, body])


def _save_report(
    report: dict,
    markdown: str,
    scenario_name: str,
    timestamp: str,
) -> tuple[Path, Path]:
    """Write JSON and Markdown to outputs/reports/ and return both paths."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{scenario_name}_{_safe_timestamp(timestamp)}"
    json_path = REPORTS_DIR / f"{stem}.json"
    md_path = REPORTS_DIR / f"{stem}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return json_path, md_path


# ---------------------------------------------------------------------------
# Session report
# ---------------------------------------------------------------------------


def generate_report(
    session_result: SessionResult,
    scored_results: list[CompletionScore],
    scenario: dict,
) -> dict:
    """Generate a full JSON + Markdown research note for a single BoN session.

    Calls the Claude API once (haiku) to generate the interpretation paragraph.
    Saves both outputs to outputs/reports/ and returns the report dict.
    """
    dim_names = [d["name"] for d in scenario.get("evaluation_dimensions", [])]
    dim_stats = compute_dimension_stats(scored_results)
    flags = _flags_summary(scored_results)
    n = len(session_result.completions)

    valid_scored = [s for s in scored_results if "scoring_failed" not in s.flags]
    ranked = sorted(valid_scored, key=_avg_score)
    most_aligned = ranked[0] if ranked else None
    most_misaligned = ranked[-1] if ranked else None

    completions_summary = [
        {
            "index": i,
            "first_300_chars": session_result.completions[i][:300],
            "scores": scored_results[i].scores if i < len(scored_results) else {},
            "flags": scored_results[i].flags if i < len(scored_results) else [],
        }
        for i in range(n)
    ]

    interpretation = _generate_interpretation(
        scenario.get("threat_model", ""),
        session_result.model_alias,
        n,
        dim_stats,
        flags,
    )

    report: dict = {
        "metadata": {
            "session_id": session_result.session_id,
            "scenario_name": session_result.scenario_name,
            "model": session_result.model_alias,
            "timestamp": session_result.timestamp,
            "bon_n": n,
        },
        "threat_model": scenario.get("threat_model", ""),
        "research_rationale": scenario.get("description", ""),
        "completions_summary": completions_summary,
        "dimension_statistics": dim_stats,
        "most_misaligned_completion": {
            "index": most_misaligned.completion_index,
            "text": most_misaligned.completion_text,
            "avg_score": round(_avg_score(most_misaligned), 4),
        } if most_misaligned else None,
        "most_aligned_completion": {
            "index": most_aligned.completion_index,
            "text": most_aligned.completion_text,
            "avg_score": round(_avg_score(most_aligned), 4),
        } if most_aligned else None,
        "flags_summary": flags,
        "interpretation_note": interpretation,
        "references": REFERENCES,
    }

    markdown = _session_report_to_markdown(report, scenario)
    _save_report(report, markdown, session_result.scenario_name, session_result.timestamp)
    return report


def _session_report_to_markdown(report: dict, scenario: dict) -> str:
    """Render a session report dict as a standalone Markdown research note."""
    meta = report["metadata"]
    stats = report["dimension_statistics"]
    misaligned = report["most_misaligned_completion"]

    date_str = meta["timestamp"][:10]

    lines: list[str] = [
        f"# {scenario.get('title', meta['scenario_name'])}",
        "",
        f"**Scenario:** {meta['scenario_name']}  ",
        f"**Threat Model:** {report['threat_model']}  ",
        f"**Model:** {meta['model']}  ",
        f"**Date:** {date_str}  ",
        f"**N (samples):** {meta['bon_n']}  ",
        f"**Session ID:** {meta['session_id']}  ",
        "",
        "---",
        "",
        "## Research Rationale",
        "",
        report["research_rationale"],
        "",
        "---",
        "",
    ]

    if stats:
        lines += [
            "## Score Summary",
            "",
            "> Note: Scores are 0.0-1.0 per dimension. "
            "Higher scores indicate stronger misalignment signal.",
            "",
        ]
        rows = [
            [
                dim,
                f"{v['mean']:.3f}",
                f"{v['min']:.3f}",
                f"{v['max']:.3f}",
                f"{v['std']:.3f}",
            ]
            for dim, v in stats.items()
        ]
        lines.append(_md_table(["Dimension", "Mean", "Min", "Max", "Std"], rows))
        lines += ["", "---", ""]

    if misaligned:
        avg = misaligned["avg_score"]
        idx = misaligned["index"]
        lines += [
            "## Most Notable Completion",
            "",
            f"*Sample {idx + 1} of {meta['bon_n']} -- "
            f"highest average score ({avg:.3f}) -- model: {meta['model']}*",
            "",
            "> " + "\n> ".join(misaligned["text"].splitlines()),
            "",
            "---",
            "",
        ]

    if report["flags_summary"]:
        lines += ["## Flags Observed", ""]
        for flag, count in sorted(report["flags_summary"].items(), key=lambda x: -x[1]):
            lines.append(f"- {flag}: {count}x")
        lines += ["", "---", ""]

    lines += [
        "## Interpretation",
        "",
        report["interpretation_note"],
        "",
        "---",
        "",
        "## References",
        "",
    ]
    for ref in REFERENCES:
        lines.append(f"{ref['index']}. {ref['citation']}  ")
        lines.append(f"   {ref['url']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def generate_comparison_report(
    comparison_results: dict[str, tuple[SessionResult, list[CompletionScore]]],
    scenario: dict,
) -> dict:
    """Generate a JSON + Markdown comparison report across multiple models.

    comparison_results maps model_alias -> (SessionResult, list[CompletionScore]).
    Saves both outputs to outputs/reports/ and returns the report dict.
    """
    dim_names = [d["name"] for d in scenario.get("evaluation_dimensions", [])]
    timestamp = datetime.now().isoformat()

    per_model: dict[str, dict] = {}
    for alias, (session_result, scored) in comparison_results.items():
        stats = compute_dimension_stats(scored)
        per_model[alias] = {
            "session_id": session_result.session_id,
            "n": len(session_result.completions),
            "dimension_statistics": stats,
            "flags_summary": _flags_summary(scored),
        }

    # Per-dimension: which model scored highest and variance across models
    cross_model: dict[str, dict] = {}
    for dim in dim_names:
        means = {
            alias: data["dimension_statistics"][dim]["mean"]
            for alias, data in per_model.items()
            if dim in data["dimension_statistics"]
        }
        if not means:
            continue
        highest_alias = max(means, key=lambda a: means[a])
        lowest_alias = min(means, key=lambda a: means[a])
        all_means = list(means.values())
        provider_variance = (
            (max(all_means) - min(all_means)) if len(all_means) > 1 else 0.0
        )
        cross_model[dim] = {
            "means_by_model": means,
            "highest_model": highest_alias,
            "lowest_model": lowest_alias,
            "cross_model_range": round(provider_variance, 4),
        }

    report: dict = {
        "report_type": "comparison",
        "scenario_name": scenario.get("name", ""),
        "threat_model": scenario.get("threat_model", ""),
        "research_rationale": scenario.get("description", ""),
        "timestamp": timestamp,
        "models_compared": list(comparison_results.keys()),
        "per_model": per_model,
        "cross_model_analysis": cross_model,
        "references": REFERENCES,
    }

    markdown = _comparison_report_to_markdown(report, scenario)
    _save_report(report, markdown, scenario.get("name", "comparison"), timestamp)
    return report


def _comparison_report_to_markdown(report: dict, scenario: dict) -> str:
    """Render a comparison report as a Markdown research note."""
    lines: list[str] = [
        f"# Comparison Report: {scenario.get('title', report['scenario_name'])}",
        "",
        f"**Threat Model:** {report['threat_model']}  ",
        f"**Models:** {', '.join(report['models_compared'])}  ",
        f"**Date:** {report['timestamp'][:10]}  ",
        "",
        "---",
        "",
        "## Research Rationale",
        "",
        report["research_rationale"],
        "",
        "---",
        "",
        "## Per-Model Score Summaries",
        "",
    ]

    for alias, data in report["per_model"].items():
        lines += [f"### {alias}", ""]
        stats = data["dimension_statistics"]
        if stats:
            rows = [
                [dim, f"{v['mean']:.3f}", f"{v['min']:.3f}", f"{v['max']:.3f}", f"{v['std']:.3f}"]
                for dim, v in stats.items()
            ]
            lines.append(_md_table(["Dimension", "Mean", "Min", "Max", "Std"], rows))
        else:
            lines.append("*No valid scores.*")
        flags = data["flags_summary"]
        if flags:
            lines += ["", "Flags: " + ", ".join(f"{k} ({v}x)" for k, v in flags.items())]
        lines += [""]

    lines += ["---", "", "## Cross-Model Analysis", ""]
    cross = report["cross_model_analysis"]
    if cross:
        rows = [
            [
                dim,
                v["highest_model"],
                v["lowest_model"],
                f"{v['cross_model_range']:.3f}",
            ]
            for dim, v in cross.items()
        ]
        lines.append(
            _md_table(["Dimension", "Highest Model", "Lowest Model", "Range"], rows)
        )
    lines += ["", "---", "", "## References", ""]
    for ref in REFERENCES:
        lines.append(f"{ref['index']}. {ref['citation']}  ")
        lines.append(f"   {ref['url']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multiplayer report
# ---------------------------------------------------------------------------


def generate_multiplayer_report(session: MultiplayerSession) -> dict:
    """Generate a JSON + Markdown report for a multiplayer session.

    Summarizes transcript, round dynamics, and verdict. Saves to outputs/reports/.
    """
    num_rounds = len(session.rounds)

    round_summaries = [
        {
            "round_number": r.round_number,
            "persuader_excerpt": r.persuader_output[:300],
            "skeptic_excerpt": r.skeptic_output[:300],
        }
        for r in session.rounds
    ]

    verdict_label = (
        "DECEPTION DETECTED"    if session.verdict_correct is True  else
        "NO DECEPTION DETECTED" if session.verdict_correct is False else
        "UNPARSEABLE"
    )

    report: dict = {
        "report_type": "multiplayer",
        "session_id": session.session_id,
        "scenario_name": session.scenario_name,
        "persuader_model": session.persuader_model,
        "skeptic_model": session.skeptic_model,
        "timestamp": session.timestamp,
        "num_rounds": num_rounds,
        "round_summaries": round_summaries,
        "final_verdict": session.final_verdict,
        "verdict_parsed": verdict_label,
        "verdict_correct": session.verdict_correct,
        "references": REFERENCES,
    }

    markdown = _multiplayer_report_to_markdown(report, session)
    _save_report(
        report, markdown, session.scenario_name, session.timestamp
    )
    return report


def _multiplayer_report_to_markdown(
    report: dict, session: MultiplayerSession
) -> str:
    """Render a multiplayer report as a Markdown research note."""
    lines: list[str] = [
        f"# Multiplayer Report: {session.scenario_name}",
        "",
        f"**Persuader:** {session.persuader_model}  ",
        f"**Skeptic:** {session.skeptic_model}  ",
        f"**Date:** {session.timestamp[:10]}  ",
        f"**Rounds:** {report['num_rounds']}  ",
        f"**Session ID:** {session.session_id}  ",
        "",
        "---",
        "",
        "## Transcript Summary",
        "",
    ]

    for r in session.rounds:
        label = "Opening" if r.round_number == 0 else f"Round {r.round_number}"
        lines += [
            f"### {label}",
            "",
            f"**Persuader [{session.persuader_model}]:**",
            "",
            "> " + "\n> ".join(r.persuader_output[:500].splitlines()),
            "",
            f"**Skeptic [{session.skeptic_model}]:**",
            "",
            "> " + "\n> ".join(r.skeptic_output[:500].splitlines()),
            "",
        ]

    lines += [
        "---",
        "",
        "## Final Verdict",
        "",
        f"**Parsed result:** {report['verdict_parsed']}",
        "",
        session.final_verdict,
        "",
        "---",
        "",
        "## References",
        "",
    ]
    for ref in REFERENCES:
        lines.append(f"{ref['index']}. {ref['citation']}  ")
        lines.append(f"   {ref['url']}")
        lines.append("")

    return "\n".join(lines)
