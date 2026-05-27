# evals

Run-able regression evals for the skills in this repo. Each case runs the skill twice — once with the skill loaded, once without — then grades both with:

1. **Deterministic assertions** on the parsed output (goal counts, work/personal split, theme present, accountability set up)
2. **Blind A/B LLM judge** comparing the two outputs across qualitative dimensions defined per skill

## Setup

```bash
pip install -r evals/requirements.txt
export ANTHROPIC_API_KEY=...
```

## Run

```bash
python evals/run.py                            # all skills, all cases
python evals/run.py --skill goal-decomposer    # one skill
python evals/run.py --case scattered-ideas     # one case
python evals/run.py --no-judge                 # skip the LLM judge (faster, cheaper)
python evals/run.py --dry-run                  # validate cases without API calls
```

Results are written to `evals/results/<timestamp>/<skill>/` (gitignored). Each case directory holds `with_skill.json`, `baseline.json`, and `judge.json`. A per-skill `summary.json` + `summary.md` are produced at the top of each run.

## Layout

```
evals/
  run.py
  requirements.txt
  cases/
    <skill-name>/
      _skill.json        # skill path, assertions config, judge rubric, force_prompt
      <case>.json        # initial_prompt + scripted_user_replies
  results/<timestamp>/   # gitignored
```

`_skill.json` is the per-skill config — what assertions to run and what dimensions the judge scores. Adding a new skill means adding a new directory under `cases/` with a `_skill.json` plus one JSON per case.

## How a case is run

Each case is a scripted multi-turn conversation. The runner walks through `initial_prompt`, the `scripted_user_replies` (one user turn each, with the assistant answering between them), then sends a `force_prompt` from `_skill.json` to coerce the final structured output. That final assistant turn is what gets parsed and graded.

The structure parser is a separate Haiku call that converts the natural-language output into `{theme, work_goals[], personal_goals[], accountability}` — keeping assertions deterministic without forcing the skill itself to emit JSON.

## Models

| Stage   | Model |
| ------- | ----- |
| Run     | `claude-sonnet-4-6` |
| Parse   | `claude-haiku-4-5-20251001` |
| Judge   | `claude-opus-4-7` |

Override by editing the constants at the top of `run.py`.
