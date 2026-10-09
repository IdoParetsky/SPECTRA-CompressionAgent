# CLAUDE.md: SPECTRA-CompressionAgent

Two agents share this working tree:
- **Claude Code is the science agent.** It does design, development, analysis and literature. Use Opus 5.5 MAX or Fable 5.1 MAX for science work, and Sonnet for implementation and monitoring.
- **The Cursor ops agent (Grok 4.6 High Effort) is separate.** It polls the BGU cluster, briefs Ido and keeps GPUs busy with pre-authorised cells. It follows `.cursor/rules/*.mdc` and `docs/OPS_HANDOFF_RUNBOOK.md` §10. If you are that agent, this file adds nothing to your instructions.

## Standing instructions for every sitting (Ido, 9 Oct 2026)
`/spectra-start` and `/spectra-thesis-mission` are standing instructions, not optional commands. Run both at the start of every sitting, and again after a pause or a context compaction, whether or not Ido types them:

1. **`/spectra-start`**: invoke the `spectra-start` skill. If it is not available (a session started outside this folder does not load `.claude/`), read `.claude/skills/spectra-start/SKILL.md` and follow its checklist by hand. It reads, in order: the top stamps of `docs/AGENT_MAIL.md`; `docs/CLAUDE_RESEARCH_HANDOFF.md` §0, §9, §11.3, §12; the top OPS DELTA; the newest queue rows; the ledger headings; and the live cluster (`squeue`, `sacct`) before any document's list of jobs.
2. **`/spectra-thesis-mission`**: invoke the `spectra-thesis-mission` skill, or read `.claude/skills/spectra-thesis-mission/SKILL.md` in full. It covers identity, NEON lineage, Gilad's frame and the state of mind.
3. **Rules.** The native stance is in `.claude/rules/` (loaded automatically). Cursor's `.cursor/rules/*.mdc` are **not** loaded here; read them only if a rule is in doubt. Where they disagree, `docs/OPS_HANDOFF_RUNBOOK.md` §10 wins. Do not ingest the ledger in full.

Open this folder (`C:\SPECTRA-CompressionAgent`) as the workspace before starting Claude Code, so that this file, `.claude/rules/` and both skills load.

## Never
- Train, or install CPU PyTorch, on this laptop. GPU work runs only on BGU Slurm (`ssh bgu-slurm`). Run remote scripts with `powershell -NoProfile -File scripts/rexec.ps1 -File <script.sh>`.
- Run `git stash`, `git checkout -- <file>`, `git restore`, `git reset --hard` or `git clean`. Each one destroys the ops agent's unstaged edits.
- Stage or commit files you did not change. Commit only your own files or hunks, then push.
- Commit `runs/`, `scripts/_tmp_*`, `NAPv2-main.zip`, Ido's personal documents (handoff §9.1) or any secret.
- Submit a cell before it is registered in `docs/SITTING_GPU_QUEUE.md`.
- Start a new train without Ido's GO, or cancel one without his explicit ask.
- Quote anything but TEST as a result (5k, with val, 10k and honest beside it), or pick a point on test.
