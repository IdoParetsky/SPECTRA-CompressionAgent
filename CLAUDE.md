# CLAUDE.md: SPECTRA-CompressionAgent

**Claude Code is the sole SPECTRA agent (Ido, 15 Nov 2026).** It does design, development, analysis, literature, cluster operations (polling, submits of registered cells, preemption handling) and Ido's briefs (the review page and pushes). Opus 5.5 MAX does the science, Fable 5.1 MAX the hardest bounded development, and Sonnet specified implementation (table below).
- **The Cursor ops agent (Grok) was retired on 15 Nov.** Nothing it ran is left: no cron and no cluster process. Its pre-authorised actions (`docs/OPS_HANDOFF_RUNBOOK.md` §10.3) concern finished September jobs. The runbook's never-lists still bind.
- **Ops' unstaged edits stay in the tree until Ido decides** (`docs/PROMPT_FABLE_V6.md`, `docs/NEXT_DEV_PHASE.md`, `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`, lines of `docs/SITTING_GPU_QUEUE.md`, `docs/paper/GILAD_NEWS_30SEP.md`, `docs/paper/GILAD_OCT8_TRACKER.md`, `.gitignore`, `.vscode/settings.json`).
- **`docs/AGENT_MAIL.md` is retired** (its last stamp tells ops to stop). The record is the queue row, the ledger, the draft and the skeleton; Ido is briefed by the review page and by pushes.

## Standing instructions for every sitting (Ido, 9 Oct 2026)
`/spectra-start` and `/spectra-thesis-mission` are standing instructions, not optional commands. Run both at the start of every sitting, and again after a pause or a context compaction, whether or not Ido types them:

1. **`/spectra-start`**: invoke the `spectra-start` skill. If it is not available (a session started outside this folder does not load `.claude/`), read `.claude/skills/spectra-start/SKILL.md` and follow its checklist by hand. It reads, in order: `docs/CLAUDE_RESEARCH_HANDOFF.md` §0 (the current-state block); the newest queue rows; `docs/paper/PAPER_SKELETON.md`; the ledger headings; and the live cluster (`squeue`, `sacct`, the managers' state files) before any document's list of jobs. The mail file and the OPS DELTA are history since 15 Nov.
2. **`/spectra-thesis-mission`**: invoke the `spectra-thesis-mission` skill, or read `.claude/skills/spectra-thesis-mission/SKILL.md` in full. It covers identity, NEON lineage, Gilad's frame and the state of mind.
3. **Rules.** The native stance is in `.claude/rules/` (loaded automatically). Cursor's `.cursor/rules/*.mdc` are ops' and history since 15 Nov. The never-lists of `docs/OPS_HANDOFF_RUNBOOK.md` §10 still bind. Do not ingest the ledger in full.

Open this folder (`C:\SPECTRA-CompressionAgent`) as the workspace before starting Claude Code, so that this file, `.claude/rules/` and both skills load.

## Models, effort and waiting (Ido, 10 Oct 2026)
| Work | Model, effort | Where it runs |
|---|---|---|
| Registrations, calls, reading results, ledger sections, reviewing every diff, design | Opus 5.5, max | the main tab |
| Hardest bounded development, deep literature synthesis | Fable 5.1, max | `spectra-hard-dev` subagent, background |
| Absorbing documents, literature scans | Opus 5.5, max | `spectra-literature` subagent, background |
| Implementing a fully written spec | Sonnet, high | `spectra-implementer` subagent; science reviews the diff; tests run on the cluster |
| Watching jobs | no model | background shell loops over `sacct` that wake the main tab once on a terminal state |

- Type `/effort max` at the start of every sitting. It lasts one session, and saved settings stop at xhigh (`.claude/settings.json` keeps xhigh as the floor). Do not set `CLAUDE_CODE_EFFORT_LEVEL`: it would force max on the Sonnet subagents too.
- Fable has its own weekly limit on Max, so treat it as a reserve. Run `/usage` at the start and at checkpoints; if Opus runs tight, use `opusplan` or pause.
- Waiting is event-driven: never poll in the main tab. No log dumps in the main tab: full outputs go to files, and the tab shows calls and compact tables.
- Since 15 Nov the session is reached from Ido's phone by Remote Control; keep the VPN watcher armed (`scripts/_tmp_s15nov_vpn_watch2.sh`) and push him when the link drops or something needs him.

## The paper draft (Ido, 14 Oct 2026)
Science owns `docs/paper/SPECTRA_draft.md` under a standing GO.
- Keep §3–§6 (Approach, Evaluation, Results, Discussion) current with the method of record.
- After each ledger section, update the draft in one reviewable commit.
- Follow the draft's conventions:
  - NEON section order;
  - numbers only by ledger reference (§N), TEST 5k with val, 10k and honest beside it;
  - every table marked PRELIM until it is locked;
  - no train-log values.
- Ops' earlier one-line pins were folded into the body on 14 Oct. Keep `docs/paper/PAPER_SKELETON.md` (the outline and table map) current with the draft.

## Never
- Train, or install CPU PyTorch, on this laptop. GPU work runs only on BGU Slurm (`ssh bgu-slurm`). Run remote scripts with `powershell -NoProfile -File scripts/rexec.ps1 -File <script.sh>`.
- Run `git stash`, `git checkout -- <file>`, `git restore`, `git reset --hard` or `git clean`. Each one destroys the ops agent's unstaged edits.
- Stage or commit files you did not change. Commit only your own files or hunks, then push.
- Commit `runs/`, `scripts/_tmp_*`, `NAPv2-main.zip`, Ido's personal documents (handoff §9.1) or any secret.
- Submit a cell before it is registered in `docs/SITTING_GPU_QUEUE.md`.
- Start a new train without Ido's GO, or cancel one without his explicit ask. Exception (Ido, 15 Nov): a train that science judges first priority needs no separate GO; register it first and tell him about every major run.
- Quote anything but TEST as a result (5k, with val, 10k and honest beside it), or pick a point on test.
