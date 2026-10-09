# Tooling on this Windows laptop (learned 9 Oct 2026)

**Git from PowerShell 5.1.** It mangles double quotes in native arguments, so `git commit -m "..."` with quotes fails into bogus pathspecs. Write the message to a file and run `git commit -F <file>`. A non-zero exit after `git push` is usually stderr progress; confirm with `git ls-remote origin refs/heads/master` against `git rev-parse HEAD`.

**Line endings.** `core.autocrlf=true`: shared docs are CRLF in the working tree (queue, runbook, mail, `CLAUDE.md`, tracker, `src/`); the ledger is LF. After an edit, count bare LF in a CRLF file and keep it at 0.

**Your own hunks only, without `git add -p`** (interactive commands are unavailable here). When the other agent has unstaged edits in a file you must commit, stage a constructed blob. Take HEAD's version plus only your hunk, in LF, `git hash-object -w <that file>`, then `git update-index --cacheinfo 100644,<sha>,<path>`. Check `git diff --cached` before committing. The working tree keeps both agents' edits.

**Cluster.**
- Scripts run with `powershell -NoProfile -File scripts/rexec.ps1 -File <script.sh>`. Filter its noise with `Select-String -NotMatch "NativeCommandError|CategoryInfo|FullyQualifiedErrorId|At C:|^\+ |^\s*$"`.
- From Git Bash, use `ssh -o BatchMode=yes -o LogLevel=ERROR bgu-slurm '<cmd>'`.
- Uploads go by `scp <file> bgu-slurm:/home/paretsky/scratch_audit/_up_<name>`. Deploy scripts strip CR and check each upload's `git hash-object` against the local blob before overlaying a new tree.

**Waiting.** Never foreground-sleep or poll in a loop in the main context. Wait with a background Bash until-loop over `sacct` that exits on a terminal state (COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, PREEMPTED); it wakes the session once.

**Delegation.**
- Sonnet subagents implement fully specified changes and poll.
- Opus or Fable subagents at max effort do heavy development, literature, and long reads that should not fill the main context.
- Review every subagent diff before it is deployed.
- A subagent edits only the files named in its brief, runs no state-changing git, and never installs or imports torch on this laptop. CPU tests run on a staged copy on the login node.
