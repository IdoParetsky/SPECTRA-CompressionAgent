# Tooling on this Windows laptop (learned 9 Oct 2026)

**Git from PowerShell 5.1.** It mangles double quotes in native arguments, so `git commit -m "..."` with quotes fails into bogus pathspecs. Write the message to a file and run `git commit -F <file>`. A non-zero exit after `git push` is usually stderr progress; confirm with `git ls-remote origin refs/heads/master` against `git rev-parse HEAD`.

**Line endings.** `core.autocrlf=true`: shared docs are CRLF in the working tree (queue, runbook, mail, `CLAUDE.md`, tracker, `src/`); the ledger is LF. After an edit, count bare LF in a CRLF file and keep it at 0.

**Your own hunks only, without `git add -p`** (interactive commands are unavailable here). When the other agent has unstaged edits in a file you must commit, stage a constructed blob. Take HEAD's version plus only your hunk, in LF, `git hash-object -w <that file>`, then `git update-index --cacheinfo 100644,<sha>,<path>`. Check `git diff --cached` before committing. The working tree keeps both agents' edits.

**Cluster.**
- Scripts run with `powershell -NoProfile -File scripts/rexec.ps1 -File <script.sh>`. Filter its noise with `Select-String -NotMatch "NativeCommandError|CategoryInfo|FullyQualifiedErrorId|At C:|^\+ |^\s*$"`.
- From Git Bash, use `ssh -o BatchMode=yes -o LogLevel=ERROR bgu-slurm '<cmd>'`.
- Uploads go by `scp <file> bgu-slurm:/home/paretsky/scratch_audit/_up_<name>`. Deploy scripts strip CR and check each upload's `git hash-object` against the local blob before overlaying a new tree.
- CPU tests on the login node: the user slice is capped at 4 CPUs and 8 GB, so set `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`, and run anything over a minute detached (`nohup … &`, output to a file); an interactive ssh session can drop mid-run (T2's tests, 10 Oct). Run one test file per pytest process, under `setsid nohup`: a combined run of eleven files died silently after about 4 min with no OOM and no exit code (10 Oct 18:27), while the same files passed one per process.

**Write files safely (learned 10 Oct 17:15).** Compute a file's new content in full before opening it for writing, then write a temp file and `os.replace` it over the target. `open(p, "wb").write(f(x))` truncates `p` before `f` runs, so an exception in `f` leaves the file empty. That emptied the shared queue and lost ops' unstaged stamps. Assert every anchor before any write.

**Disk quota (learned 11–12 Oct).** `/home/paretsky` is NFS with a per-user quota that `quota` does not show. Use sits close to it (about 127 GB on 12 Oct), so probe with a small write before a wave. A job that starts over quota dies in seconds with exit `0:53` and no log, or finishes COMPLETED without its rows. NFS can report `Disk quota exceeded` for a write that still lands. `scripts/_tmp_s12oct_watch.sh` exits 6 on a failed write. Snapshot directories (`traj_models`) are the bulk that can be freed: back them up first (Drive once Ido sets up an OAuth remote, else the laptop) and delete only after a per-file check.

**Backslashes in shell heredocs (learned 12 Oct).** Through this Bash tool, even a quoted heredoc turns `\\` into `\`, so a Python literal `"C:\\x\\tree"` arrives as `"C:\x\tree"` and `\t` becomes a tab. Write any script that holds a Windows path with the Write tool.

**Waiting is event-driven (Ido, 10 Oct).** Never foreground-sleep or poll in a loop in the main tab. Wait with a background Bash until-loop over `sacct` that exits on a terminal state (COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, PREEMPTED); it wakes the session once. Let it exit early on the first non-COMPLETED terminal state, so a failure is seen at once. A timed wake (a background `sleep` to a clock time) is fine for a promised report. A watcher must treat a dead link as an event: exit after about four consecutive polls that cannot reach the cluster (`scripts/_tmp_s11oct_watch.sh`, exit 5), say so in the brief and the mail, and then wait on a link check. The BGU VPN dropped twice on 11 Oct, and a watcher that retried silently lost a working day of reads.

**Smokes and priorities (learned 10 Oct).** A smoke that exits 0 can still have failed: the runner catches a per-network exception and finishes with status 0, so afterok releases the dependents (T0-F2's first eval smoke). Read each smoke's start-check lines before any cell runs: submit the cells held (`scontrol hold`) and release them after the read. Slurm's age factor, counted from eligibility, outweighs nice gaps of a few units, so nice alone does not order a long queue; re-nice or hold explicitly. `SBATCH_HOLD=1` does not hold jobs submitted through `scripts/submit.sh` (HEAD-F, 12 Oct): run `scontrol hold <id>` right after each submit.

**No log dumps in the main tab (Ido, 10 Oct).** Readers write their full output to a scratchpad file; the tab shows the call lines and compact tables only. Never paste a raw log, a whole reader output or a subagent transcript.

**Delegation** (agent files in `.claude/agents/`; the table is in `CLAUDE.md`).
- `spectra-implementer` (Sonnet, high) implements fully specified changes.
- `spectra-literature` (Opus, max) absorbs documents and runs literature scans. `spectra-hard-dev` (Fable 5.1, max; its own weekly limit, a reserve) does the hardest bounded development and deep synthesis.
- Monitoring takes no model: background shell loops.
- Review every subagent diff before it is deployed.
- A subagent edits only the files named in its brief, runs no state-changing git, and never installs or imports torch on this laptop. CPU tests run on a staged copy on the login node.
