# CLAUDE.md: SPECTRA-CompressionAgent

Two agents share this working tree:
- **Claude Code is the science agent.** It does design, development, analysis and literature. Use Opus 5.5 MAX or Fable 5.1 MAX for science work, and Sonnet for implementation and monitoring.
- **The Cursor ops agent (Grok 4.6 High Effort) is separate.** It polls the BGU cluster, briefs Ido and keeps GPUs busy with pre-authorised cells. It follows `.cursor/rules/*.mdc` and `docs/OPS_HANDOFF_RUNBOOK.md` §10. If you are that agent, this file adds nothing to your instructions.

## At the start of every Claude Code session
1. Read `docs/CLAUDE_RESEARCH_HANDOFF.md`, starting with §0, §9, §11.3 and §12. Its §13 is the launch brief.
2. Read the five rule files in `.cursor/rules/`. Cursor loads them automatically; Claude Code does not. Where they disagree, `docs/OPS_HANDOFF_RUNBOOK.md` §10 wins.
3. Check the live cluster state (`squeue`, `sacct`) before trusting any document's list of jobs.

## Never
- Train, or install CPU PyTorch, on this laptop. GPU work runs only on BGU Slurm (`ssh bgu-slurm`). Run remote scripts with `powershell -NoProfile -File scripts/rexec.ps1 -File <script.sh>`.
- Run `git stash`, `git checkout -- <file>`, `git restore`, `git reset --hard` or `git clean`. Each one destroys the ops agent's unstaged edits.
- Stage or commit files you did not change. Commit only your own files or hunks, then push.
- Commit `runs/`, `scripts/_tmp_*`, `NAPv2-main.zip`, Ido's personal documents (handoff §9.1) or any secret.
- Submit a cell before it is registered in `docs/SITTING_GPU_QUEUE.md`.
- Start a new train without Ido's GO, or cancel one without his explicit ask.
- Quote anything but TEST as a result (5k, with val, 10k and honest beside it), or pick a point on test.
