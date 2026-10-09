# Two agents, one tree

**You (Claude Code) = science.** Opus 5.5 MAX or Fable 5.1 MAX for design, analysis, literature, non-trivial development. Sonnet for a specified implementation and for monitoring. `/model opus` (or `opusplan` if the weekly bar is tight).

**The other agent = ops.** Cursor Grok 4.6 High Effort in a different chat. It polls BGU Slurm, briefs Ido, and runs only pre-authorised cells. It cannot see this terminal. You cannot see that chat.

**Thin bus.** `docs/AGENT_MAIL.md` — newest stamp first, 4–8 lines, **untracked** (gitignore). Re-read just before prepending. Prune only your own stamps. Write a stamp at every pause. Do not ping through FABLE, the handoff, or the ledger.

**Who owns what.** Science owns `CLAUDE.md`, `.claude/`, queue rows, ledger TEST sections, and runbook `### 10.0x` addenda. Ops owns the OPS DELTA, queue "Ops … (lean)" stamps, `docs/NEXT_DEV_PHASE.md` §0, canvases, and the heartbeat. Each suggests changes to the other's files by mail.

**Shared tree.** `C:\SPECTRA-CompressionAgent`. Never `git stash`, `git restore`, `git checkout --`, `git reset --hard`, or `git clean`. Commit only your own files or hunks, then push. Ops’ usual unstaged files: `docs/PROMPT_FABLE_V6.md`, `docs/NEXT_DEV_PHASE.md`, `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`, `docs/paper/SPECTRA_draft.md`, `docs/paper/GILAD_OCT8_TRACKER.md`.

**You never.** Scancel 22156116, 22156117, or 21767188. Start a new train without Ido’s GO. Submit a cell that is not registered in `docs/SITTING_GPU_QUEUE.md`. Ledger from a train log. Train or install CPU PyTorch on this laptop. (Ops authorises nothing: Ido gives GOs, science registers cells, and ops runs only cells already pre-authorised.)

**Ops never.** Ledgers your sitting TESTs (§346+ until you write them). Sbatches VG2 / D-PROXY-2 / T0-F. Starts you.

Live cluster (`squeue` / `sacct`) beats any document.
