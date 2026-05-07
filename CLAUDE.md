# CLAUDE.md -- DeceptionScope

This file is the primary context document for Claude Code sessions on this project.
Read it fully before making any changes to files.

---

## Project Purpose

DeceptionScope is an alignment research tool for probing how language models behave
when placed in structured scenarios designed to elicit deceptive, sycophantic, or
situationally-aware behavior. It implements the model organisms of misalignment
methodology from Hubinger et al. (2023) using Best-of-N (BoN) sampling across
multiple model providers, a Flask web UI, a Rich terminal dashboard, and a two-model
multiplayer adversarial runner.

No models are trained or fine-tuned. The tool probes behavioral patterns using
carefully constructed system prompts and scores outputs using a separate Claude API call.

---

## Key Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Copy and fill in API keys:
```bash
cp .env.example .env
# Then edit .env with your ANTHROPIC_API_KEY (required) and optionally
# OPENAI_API_KEY and GOOGLE_AI_API_KEY
```

List all available scenarios:
```bash
python main.py list
```

Run a single BoN probe session:
```bash
python main.py run sycophancy_probe
python main.py run sycophancy_probe --model gpt4o-mini --n 5
python main.py run sycophancy_probe --no-dashboard
```

Run a cross-model comparison:
```bash
python main.py compare situational_awareness --models haiku,gpt4o-mini,gemini-flash
```

Run a multiplayer (persuader vs. skeptic) session:
```bash
python main.py multiplayer multiplayer_debate --persuader haiku --skeptic gpt4o-mini
```

Re-generate a report from a saved session:
```bash
python main.py report <session_id>
```

Start the web UI:
```bash
python main.py serve
# Then open http://localhost:5000
```

Run tests:
```bash
python -m pytest tests/
```

---

## Architecture

### src/models.py
Provider abstraction layer. Defines ModelProvider enum, ModelConfig dataclass, and
the AVAILABLE_MODELS registry. The central function is complete(), which accepts a
system prompt, user turns, and a model alias and returns a plain string completion.
Supports Anthropic (claude-haiku-3-5, claude-sonnet-4-5), OpenAI (gpt-4o-mini,
gpt-4o), and Google (gemini-1.5-flash, gemini-1.5-pro). Missing optional provider
keys do not crash on import -- they raise ProviderError only when complete() is called.

### src/runner.py
Best-of-N sampling engine. run_bon_session() calls complete() N times independently
(sequentially, not in parallel) and returns a SessionResult dataclass. Saves results
to outputs/sessions/ as JSON. run_comparison_session() runs the same scenario against
multiple model aliases and returns a dict of SessionResult keyed by alias.

### src/scorer.py
Multi-dimension scorer. Always uses claude-haiku-3-5 for scoring, regardless of which
model was probed. score_completion() sends a structured scoring prompt and parses the
JSON response. Returns CompletionScore with per-dimension float scores (0.0-1.0),
a flags list, and a reasoning string. compute_dimension_stats() returns mean/min/max/std
across all N completions. Handles JSON parse failures gracefully with a "scoring_failed"
flag rather than raising an exception.

### src/multiplayer.py
Two-model adversarial runner. run_multiplayer() runs a structured N-round conversation
where one model plays "persuader" and another plays "skeptic." After the final round,
the skeptic produces a verdict. Saves the full transcript as MultiplayerSession JSON.
run_tournament() runs all combinations of model pairs and summarizes detection rates.
Framed as a debate and scalable oversight exercise (Irving et al., 2018).

### src/scenario_loader.py
YAML loader and validator. Raises descriptive ValueError for any missing required field.
list_scenarios() prints a summary table via rich. get_multiplayer_scenarios() filters
to scenarios with persuader_system_prompt and skeptic_system_prompt fields.

### src/dashboard.py
Rich terminal dashboard. Used during CLI runs. Shows live completions with ASCII color
indicators ([SIGNAL] / [ALIGNED] / [?] for red/green/yellow), a post-session score
summary table, and a flags section. No emojis. Uses rich.live, rich.table, rich.panel.

### src/report.py
Report generator. Produces both JSON and Markdown reports. The Markdown output is
formatted as a standalone research note suitable for sharing. Generates an
interpretation paragraph via Claude API. Handles single-model sessions, comparison
sessions, and multiplayer sessions with separate generator functions.

### web/app.py
Flask REST server. Serves all scenario, model, session, and report data as JSON.
Exposes POST endpoints for running sessions and scoring. Serves the static HTML/CSS/JS
frontend. Uses flask-cors for development convenience.

### web/templates/index.html + web/static/style.css + web/static/main.js
Editorial-style web frontend. IBM Plex Serif and IBM Plex Mono fonts. Off-white
background, dark ink, accent red for active/high-score states. Three tabs:
"Run Probe", "Compare Models", "Multiplayer". Vanilla JavaScript only (no frameworks).
Chart.js loaded from CDN for comparison bar charts. All styles in style.css -- no
inline styles in HTML.

### main.py
Typer CLI with commands: run, compare, multiplayer, serve, list, report.

---

## Scenario YAML Schema

```yaml
name: string                    # Short snake_case identifier
title: string                   # Human-readable title
threat_model: string            # Alignment failure category
description: string             # 1-2 sentence research rationale
system_prompt: string           # System prompt for the probed model
user_turns:
  - string                      # 1-3 sequential user messages
evaluation_dimensions:
  - name: string
    description: string         # What 0.0-1.0 means for this dimension
bon_n: int                      # Number of independent samples (5-10)
tags:
  - string
```

For multiplayer scenarios, add these fields instead of (or in addition to) user_turns:
```yaml
persuader_system_prompt: string
skeptic_system_prompt: string
opening_statement: string
num_rounds: int
```

---

## Available Model Aliases

| Alias | Model ID | Provider | Approx cost per 1k tokens |
|---|---|---|---|
| haiku | claude-haiku-3-5 | Anthropic | $0.0008 |
| sonnet | claude-sonnet-4-5 | Anthropic | $0.003 |
| gpt4o-mini | gpt-4o-mini | OpenAI | $0.00015 |
| gpt4o | gpt-4o | OpenAI | $0.005 |
| gemini-flash | gemini-1.5-flash | Google | $0.000075 |
| gemini-pro | gemini-1.5-pro | Google | $0.00125 |

The scorer always uses "haiku" regardless of which model is being probed.
Cost estimates in the terminal dashboard use cost_per_1k_tokens from ModelConfig.

---

## Adding a New Scenario

1. Create a new .yaml file in scenarios/ following the schema above.
2. Run python main.py list to verify it loads without validation errors.
3. Run python main.py run <your_scenario_name> --n 3 for a quick smoke test.

No code changes are required to add scenarios.

---

## Output Files

All outputs are written to the outputs/ directory, which is gitignored.

outputs/sessions/{scenario_name}_{timestamp}.json     -- raw SessionResult
outputs/sessions/{session_id}_scored.json             -- scored CompletionScore list
outputs/sessions/multiplayer_{session_id}.json        -- MultiplayerSession
outputs/sessions/comparison_{scenario}_{timestamp}.json -- comparison session

outputs/reports/{scenario_name}_{timestamp}.json      -- full report dict
outputs/reports/{scenario_name}_{timestamp}.md        -- Markdown research note

---

## Coding Conventions

- All source files use type hints.
- All public functions have docstrings.
- No print() statements in src/ -- use rich.console for any terminal output.
- JSON files are saved with indent=2.
- All API calls go through complete() in src/models.py -- do not import provider
  SDKs directly in other modules.
- Scorer always uses claude-haiku-3-5. Do not make this configurable to avoid
  score inconsistency across sessions.
- No emojis anywhere in terminal output or CLI messages. Use plain ASCII indicators.
- No em dashes in any generated text or documentation.

---

## Research Context

This project implements the model organisms of misalignment methodology from:

Hubinger, E., Perez, E., Schiefer, N., et al. (2023). Model Organisms of Misalignment:
The Case for a New Pillar of Alignment Research. AI Alignment Forum.
https://www.alignmentforum.org/posts/ChDH335ckdvpxXaXX

The Sleeper Agents paper demonstrates this methodology empirically:

Hubinger, E., Denison, C., Mu, J., et al. (2024). Sleeper Agents: Training Deceptive
LLMs that Persist Through Safety Training. arXiv:2401.05566.

The multiplayer runner is inspired by:

Irving, G., Christiano, P., Amodei, D. (2018). AI Safety via Debate. arXiv:1805.00899.

Key principle: BoN sampling with structured prompts is a fast, low-cost way to probe
misalignment-adjacent behaviors without any fine-tuning. Each independent sample from
the same prompt exposes behavioral variance -- the distribution of outputs across N
samples is the measurement, not any single response.

---

## Cost Reference

Rough cost for a full suite run (all 6 scenarios, N=8, scoring, one multiplayer session,
report generation) on claude-haiku-3-5: approximately $0.04.

For comparison sessions using GPT-4o-mini: add roughly $0.01 per session.
Gemini Flash is the cheapest option at $0.000075 per 1k tokens.

---

## Safety Notes

All built-in scenarios probe behavioral and rhetorical patterns only. No scenario
instructs a model to produce dangerous, harmful, or illegal content. Every output
passes through provider safety filters. The deception being studied is role-consistency
and compliance under instruction -- the same class of behavior studied in the published
literature cited above. All scenario content is publishable without concern.

---

Project: DeceptionScope
Author: Jake
Affiliation: UCSD Physics and Data Science (graduating June 2026); UC Berkeley MIDS (Fall 2026)
Research context: Duarte Lab -> ThoughtTrace -> EmbeddingDrift -> DeceptionScope
Build guide version: 2.0 | May 2026
