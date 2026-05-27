"""Eval runner for skills in this repo.

Runs each test case twice — once with the skill loaded as the system prompt, once
without — then parses the final assistant output into a structured form, applies
deterministic assertions, and asks an LLM judge to compare the two outputs blind
across the dimensions declared in `_skill.json`.

Layout this script expects:

    evals/
      cases/
        <skill-name>/
          _skill.json              # skill_path, force_prompt, assertions, judge_dimensions
          <case-name>.json         # initial_prompt, scripted_user_replies
      results/<timestamp>/         # written here (gitignored)

Run:
    export ANTHROPIC_API_KEY=...
    pip install -r evals/requirements.txt
    python evals/run.py                            # all skills, all cases
    python evals/run.py --skill goal-decomposer    # filter
    python evals/run.py --case between-jobs-exploring
    python evals/run.py --dry-run                  # no API, validate cases only
    python evals/run.py --no-judge                 # skip LLM judge phase
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = Path(__file__).resolve().parent
CASES_DIR = EVALS_DIR / "cases"
RESULTS_DIR = EVALS_DIR / "results"

RUN_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-opus-4-7"
PARSER_MODEL = "claude-haiku-4-5-20251001"

MAX_TOKENS_RUN = 2048
MAX_TOKENS_PARSE = 1024
MAX_TOKENS_JUDGE = 2048


@dataclass
class SkillConfig:
    name: str
    skill_path: Path
    force_prompt: str
    assertions: dict[str, Any]
    judge_dimensions: list[dict[str, str]]


@dataclass
class Case:
    name: str
    description: str
    initial_prompt: str
    scripted_user_replies: list[str]


@dataclass
class RunOutput:
    label: str  # "with_skill" or "baseline"
    transcript: list[dict[str, str]]  # role/content turns
    final_assistant: str
    parsed: dict[str, Any] = field(default_factory=dict)
    assertions: dict[str, Any] = field(default_factory=dict)


def load_skill_config(skill_dir: Path) -> SkillConfig:
    cfg = json.loads((skill_dir / "_skill.json").read_text())
    skill_path = (skill_dir / cfg["skill_path"]).resolve()
    if not skill_path.exists():
        raise FileNotFoundError(f"skill_path not found: {skill_path}")
    return SkillConfig(
        name=cfg["name"],
        skill_path=skill_path,
        force_prompt=cfg["force_prompt"],
        assertions=cfg["assertions"],
        judge_dimensions=cfg["judge_dimensions"],
    )


def load_cases(skill_dir: Path, case_filter: str | None) -> list[Case]:
    cases: list[Case] = []
    for path in sorted(skill_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        data = json.loads(path.read_text())
        if case_filter and data["name"] != case_filter:
            continue
        cases.append(
            Case(
                name=data["name"],
                description=data.get("description", ""),
                initial_prompt=data["initial_prompt"],
                scripted_user_replies=data.get("scripted_user_replies", []),
            )
        )
    return cases


def run_conversation(
    client,
    system_prompt: str | None,
    case: Case,
    force_prompt: str,
) -> RunOutput:
    """Walk the scripted conversation and return the final assistant turn.

    Sequence: initial_prompt -> assistant -> reply[0] -> assistant -> ... -> force_prompt -> assistant.
    The final assistant turn is what gets parsed and graded.
    """
    transcript: list[dict[str, str]] = []
    messages: list[dict[str, str]] = []

    user_turns = [case.initial_prompt, *case.scripted_user_replies, force_prompt]
    last_assistant = ""

    for user_text in user_turns:
        messages.append({"role": "user", "content": user_text})
        transcript.append({"role": "user", "content": user_text})

        kwargs: dict[str, Any] = {
            "model": RUN_MODEL,
            "max_tokens": MAX_TOKENS_RUN,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        resp = client.messages.create(**kwargs)
        assistant_text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        messages.append({"role": "assistant", "content": assistant_text})
        transcript.append({"role": "assistant", "content": assistant_text})
        last_assistant = assistant_text

    return RunOutput(
        label="",  # filled by caller
        transcript=transcript,
        final_assistant=last_assistant,
    )


PARSER_SYSTEM = """You extract structured sprint plans from natural-language output.

Given the final assistant message, return a strict JSON object with this exact shape:

{
  "theme": "<one-line theme, or empty string if none>",
  "work_goals": ["...", "..."],
  "personal_goals": ["...", "..."],
  "accountability": {
    "present": true,
    "partners": "<who the user will tell, or empty>",
    "cadence": "<how often / what touchpoints, or empty>",
    "kickoff_template_present": true
  }
}

Rules:
- A "goal" is something the person commits to doing or finishing in the sprint.
- If a goal explicitly relates to job/career/business/product/project/learning toward a craft, classify it as work. If it relates to body, mind, relationships, home, habits, experiences, hobbies (not pursued as career), classify it as personal. When ambiguous, prefer the category implied by the surrounding section header.
- If the output has section headers (e.g., "Work goals:", "Personal goals:"), respect them.
- Set accountability.present = true only if a real plan (partners + cadence, or a kickoff email draft) is described. Generic advice like "find an accountability partner" without specifics is NOT present.
- kickoff_template_present = true only if there is an actual draft/template of the kickoff message, not just a description of one.
- Output ONLY the JSON object, no prose, no code fences."""


def parse_output(client, final_assistant: str) -> dict[str, Any]:
    resp = client.messages.create(
        model=PARSER_MODEL,
        max_tokens=MAX_TOKENS_PARSE,
        system=PARSER_SYSTEM,
        messages=[{"role": "user", "content": final_assistant}],
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
    # Strip code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": text[:500]}


def apply_assertions(parsed: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Run structural checks against the parsed plan."""
    work = parsed.get("work_goals") or []
    personal = parsed.get("personal_goals") or []
    total = len(work) + len(personal)
    theme = (parsed.get("theme") or "").strip()
    accountability = parsed.get("accountability") or {}

    checks = {
        "goal_count_in_range": cfg["goal_count_min"] <= total <= cfg["goal_count_max"],
        "work_goal_min_met": len(work) >= cfg["work_goal_min"],
        "personal_goal_min_met": len(personal) >= cfg["personal_goal_min"],
        "theme_present": bool(theme) if cfg.get("requires_theme") else True,
        "accountability_present": bool(accountability.get("present"))
        if cfg.get("requires_accountability")
        else True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "totals": {
            "work_goal_count": len(work),
            "personal_goal_count": len(personal),
            "total_goal_count": total,
            "theme_chars": len(theme),
        },
    }


JUDGE_SYSTEM_TEMPLATE = """You are a blind A/B judge for sprint-planning outputs.

You will see two final-turn responses (A and B) to the same user. You do not know which is which. Rate each dimension by picking A, B, or tie, and explain in one sentence what you saw.

Dimensions to score:
{dimensions}

Return a single JSON object:
{{
  "verdicts": [
    {{"dimension": "<name>", "winner": "A" | "B" | "tie", "reason": "<one short sentence>"}}
  ]
}}

Output ONLY the JSON object."""


def judge_pair(
    client,
    case: Case,
    out_a: RunOutput,
    out_b: RunOutput,
    dimensions: list[dict[str, str]],
) -> dict[str, Any]:
    dim_lines = "\n".join(f"- {d['name']}: {d['description']}" for d in dimensions)
    system = JUDGE_SYSTEM_TEMPLATE.format(dimensions=dim_lines)

    user = (
        f"User context (description): {case.description}\n\n"
        f"User's first message: {case.initial_prompt}\n\n"
        f"--- RESPONSE A ---\n{out_a.final_assistant}\n\n"
        f"--- RESPONSE B ---\n{out_b.final_assistant}\n"
    )

    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=MAX_TOKENS_JUDGE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": text[:500]}


def summarize(
    skill_name: str,
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(case_results)
    with_skill_pass = sum(1 for r in case_results if r["with_skill"]["assertions"].get("passed"))
    baseline_pass = sum(1 for r in case_results if r["baseline"]["assertions"].get("passed"))

    judge_tally: dict[str, dict[str, int]] = {}
    for r in case_results:
        judge = r.get("judge") or {}
        for verdict in judge.get("verdicts", []) or []:
            dim = verdict["dimension"]
            # Map A/B back to with_skill/baseline using the recorded mapping.
            mapping = r.get("judge_mapping", {})
            label = mapping.get(verdict["winner"], verdict["winner"])
            judge_tally.setdefault(dim, {"with_skill": 0, "baseline": 0, "tie": 0})
            judge_tally[dim][label] = judge_tally[dim].get(label, 0) + 1

    return {
        "skill": skill_name,
        "total_cases": total,
        "assertions": {
            "with_skill_pass": with_skill_pass,
            "baseline_pass": baseline_pass,
        },
        "judge_tally": judge_tally,
    }


def render_summary_md(summary: dict[str, Any], case_results: list[dict[str, Any]]) -> str:
    lines = [f"# Eval summary — {summary['skill']}", ""]
    a = summary["assertions"]
    total = summary["total_cases"]
    lines.append(f"**Cases:** {total}")
    lines.append(
        f"**Deterministic assertions:** with-skill {a['with_skill_pass']}/{total} · "
        f"baseline {a['baseline_pass']}/{total}"
    )
    lines.append("")

    if summary["judge_tally"]:
        lines.append("## Blind A/B judge")
        lines.append("")
        lines.append("| Dimension | with-skill | baseline | tie |")
        lines.append("| --- | ---: | ---: | ---: |")
        for dim, counts in summary["judge_tally"].items():
            lines.append(
                f"| {dim} | {counts.get('with_skill', 0)} | "
                f"{counts.get('baseline', 0)} | {counts.get('tie', 0)} |"
            )
        lines.append("")

    lines.append("## Per-case")
    lines.append("")
    for r in case_results:
        name = r["case"]
        ws = r["with_skill"]["assertions"]
        bl = r["baseline"]["assertions"]
        lines.append(f"### {name}")
        lines.append(
            f"- with-skill: {'PASS' if ws.get('passed') else 'FAIL'} "
            f"(goals: {ws.get('totals', {}).get('total_goal_count', '?')})"
        )
        lines.append(
            f"- baseline: {'PASS' if bl.get('passed') else 'FAIL'} "
            f"(goals: {bl.get('totals', {}).get('total_goal_count', '?')})"
        )
        if r.get("judge"):
            for v in r["judge"].get("verdicts", []):
                mapping = r.get("judge_mapping", {})
                label = mapping.get(v["winner"], v["winner"])
                lines.append(f"  - **{v['dimension']}** → {label}: {v['reason']}")
        lines.append("")

    return "\n".join(lines)


def run_skill(
    client,
    skill_dir: Path,
    case_filter: str | None,
    do_judge: bool,
    out_dir: Path,
) -> None:
    cfg = load_skill_config(skill_dir)
    cases = load_cases(skill_dir, case_filter)
    if not cases:
        print(f"  (no cases for {cfg.name})")
        return

    system_prompt = cfg.skill_path.read_text()
    skill_results_dir = out_dir / cfg.name
    skill_results_dir.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = []
    for case in cases:
        print(f"  [{cfg.name}/{case.name}] running with-skill...")
        with_skill = run_conversation(client, system_prompt, case, cfg.force_prompt)
        with_skill.label = "with_skill"

        print(f"  [{cfg.name}/{case.name}] running baseline...")
        baseline = run_conversation(client, None, case, cfg.force_prompt)
        baseline.label = "baseline"

        print(f"  [{cfg.name}/{case.name}] parsing outputs...")
        with_skill.parsed = parse_output(client, with_skill.final_assistant)
        baseline.parsed = parse_output(client, baseline.final_assistant)
        with_skill.assertions = apply_assertions(with_skill.parsed, cfg.assertions)
        baseline.assertions = apply_assertions(baseline.parsed, cfg.assertions)

        judge_result: dict[str, Any] | None = None
        judge_mapping: dict[str, str] = {}
        if do_judge:
            print(f"  [{cfg.name}/{case.name}] judging blind A/B...")
            flip = random.random() < 0.5
            out_a, out_b = (baseline, with_skill) if flip else (with_skill, baseline)
            judge_mapping = {
                "A": out_a.label,
                "B": out_b.label,
                "tie": "tie",
            }
            judge_result = judge_pair(client, case, out_a, out_b, cfg.judge_dimensions)

        case_dir = skill_results_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "with_skill.json").write_text(
            json.dumps(with_skill.__dict__, indent=2, default=str)
        )
        (case_dir / "baseline.json").write_text(
            json.dumps(baseline.__dict__, indent=2, default=str)
        )
        if judge_result is not None:
            (case_dir / "judge.json").write_text(
                json.dumps(
                    {"verdicts": judge_result.get("verdicts", []), "mapping": judge_mapping},
                    indent=2,
                )
            )

        case_results.append(
            {
                "case": case.name,
                "with_skill": with_skill.__dict__,
                "baseline": baseline.__dict__,
                "judge": judge_result,
                "judge_mapping": judge_mapping,
            }
        )

    summary = summarize(cfg.name, case_results)
    (skill_results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = render_summary_md(summary, case_results)
    (skill_results_dir / "summary.md").write_text(md)
    print()
    print(md)


def dry_run(skill_filter: str | None, case_filter: str | None) -> int:
    """Validate case files without calling the API."""
    skill_dirs = [p for p in CASES_DIR.iterdir() if p.is_dir()]
    if skill_filter:
        skill_dirs = [p for p in skill_dirs if p.name == skill_filter]
    if not skill_dirs:
        print("No skills found.")
        return 1

    errors = 0
    for skill_dir in skill_dirs:
        try:
            cfg = load_skill_config(skill_dir)
        except Exception as exc:
            print(f"[{skill_dir.name}] config error: {exc}")
            errors += 1
            continue
        cases = load_cases(skill_dir, case_filter)
        print(f"[{cfg.name}] skill_path={cfg.skill_path.relative_to(REPO_ROOT)} cases={len(cases)}")
        for case in cases:
            print(
                f"  - {case.name}: {len(case.scripted_user_replies)} scripted replies + force_prompt"
            )
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run skill evals.")
    parser.add_argument("--skill", help="Only run this skill (matches dir under evals/cases/)")
    parser.add_argument("--case", help="Only run this case (by name field)")
    parser.add_argument("--no-judge", action="store_true", help="Skip blind A/B judge")
    parser.add_argument("--dry-run", action="store_true", help="Validate cases, no API calls")
    args = parser.parse_args()

    if args.dry_run:
        return dry_run(args.skill, args.case)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    try:
        from anthropic import Anthropic
    except ImportError:
        print("anthropic package missing. Run: pip install -r evals/requirements.txt", file=sys.stderr)
        return 2

    client = Anthropic()

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {out_dir.relative_to(REPO_ROOT)}\n")

    skill_dirs = [p for p in CASES_DIR.iterdir() if p.is_dir()]
    if args.skill:
        skill_dirs = [p for p in skill_dirs if p.name == args.skill]

    for skill_dir in skill_dirs:
        print(f"== {skill_dir.name} ==")
        run_skill(
            client,
            skill_dir,
            args.case,
            do_judge=not args.no_judge,
            out_dir=out_dir,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
