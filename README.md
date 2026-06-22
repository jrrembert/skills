# skills

[![skills.sh](https://skills.sh/b/jrrembert/skills)](https://skills.sh/jrrembert/skills)

Agent skills by [@jrrembert](https://github.com/jrrembert), discoverable on [skills.sh](https://skills.sh) and ready to publish to [ClawHub](https://clawhub.ai).

## Install

From ClawHub:

```bash
clawhub install @jrrembert/goal-decomposer
```

From skills.sh:

```bash
npx skills add jrrembert/skills
```

Or install a single skill:

```bash
npx skills add jrrembert/skills/goal-decomposer
```

## Skills

### [goal-decomposer](./goal-decomposer/SKILL.md)

Turn fuzzy ambitions into practical two-week sprint plans with concrete goals, work/personal balance, and lightweight accountability.

Useful for goal-setting, life planning, side projects, job-search structure, sabbaticals, freelance planning, habit resets, personal productivity, and unstructured phases where someone needs a clear plan and accountability.

The skill produces:

- A one-line sprint theme
- 4-6 concrete goals split into work and personal goals
- A lightweight accountability setup
- Kickoff, mid-sprint, and end-of-sprint email templates

## ClawHub publishing

The repo includes a reusable GitHub Actions workflow at [`.github/workflows/clawhub-skill-publish.yml`](./.github/workflows/clawhub-skill-publish.yml).

- Pull requests run a ClawHub dry run.
- Pushes to `main` publish changed skills under the `jrrembert` owner.
- Manual runs default to dry run; uncheck `dry_run` to publish.

To enable real publishes, add a `CLAWHUB_TOKEN` repository secret. The workflow publishes each immediate skill folder under the repo root, so new skills only need their own folder with a `SKILL.md`.

Manual publish:

```bash
clawhub skill publish ./goal-decomposer --owner jrrembert
```
