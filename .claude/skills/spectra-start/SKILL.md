---
name: spectra-start
description: Boot a SPECTRA science sitting. Standing instruction (Ido, 9 Oct 2026) - run it at the start of every Claude Code sitting and again after a pause or a context compaction, whether or not Ido types /spectra-start.
---

# /spectra-start — SPECTRA science boot

You are the SPECTRA science agent in Claude Code. Ops is a different Cursor Grok chat. Do not skip this checklist. It is a standing instruction for every sitting, together with the `spectra-thesis-mission` skill.

## 0. Model, effort, usage (Ido types these; documented in code.claude.com/docs/en/model-config, checked 9 Oct)

- **Main tab: Opus 5.5** (`/model opus`, the Max default) with **`/effort max`**. Opus 5.5 starts at *medium* effort, and `/effort max` lasts one session only, so set it every sitting. Saved settings go no higher than xhigh.
- **`opusplan`** = Opus in plan mode, Sonnet for execution. Use it only when the Opus allowance is tight.
- **Fable 5.1** has its own weekly limit on Max. Keep it for the hardest bounded subagent work: design-critical development and deep literature synthesis.
- **Sonnet** subagents take specified implementation and monitoring.
- Run **`/usage`** at the start and at checkpoints. Session and weekly limits are shared across models, and parallel subagents spend quota roughly in proportion.
- Run **`/context`** to confirm that `CLAUDE.md`, `.claude/rules/` and the skills loaded. If they did not, the tab was started outside the repo: `/cd C:\SPECTRA-CompressionAgent`, or open that folder in Cursor.

## 1. Live git (already inlined below)

Status:

!`git status -sb`

Recent commits:

!`git log -8 --oneline`

If HEAD is behind origin, `git pull --ff-only` only when the index is clean of **your** work. Never stash.

## 2. Read, in this order (do not ingest the whole ledger)

1. `docs/AGENT_MAIL.md` — top stamp only.
2. `docs/CLAUDE_RESEARCH_HANDOFF.md` — §0, §9, §11.3, §12. Rest as needed.
3. Top **OPS DELTA** in `docs/PROMPT_FABLE_V6.md`.
4. The newest queue rows (the registry table near the top of `docs/SITTING_GPU_QUEUE.md`) and the newest dated cell section before `## O38`.
5. Invoke the `spectra-thesis-mission` skill (identity, NEON, Gilad frame), or read its `SKILL.md` in full. It is a standing instruction too.
6. Grep ledger headings `§330`–latest; do not read `docs/paper/RESULTS_LEDGER.md` in full.

## 3. Cluster before documents

Poll `squeue` and `sacct` via `powershell -NoProfile -File scripts/rexec.ps1 -File <script.sh>` (or equivalent). Documents lie; the queue does not.

Keep 22156116, 22156117, 21767188. Current state (what is live, unread, or next) comes from the top stamps of `docs/AGENT_MAIL.md` and the newest queue rows, not from this checklist. A job already submitted is never resubmitted without a registered reason.

## 4. Reply to Ido

Four lines: (a) model in use, (b) QOS R/PD, (c) top mail stamp, (d) the next action from handoff §9.4 that is still open. Then wait, or do that action if it is already GO’d and registered.
