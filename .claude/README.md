# Claude Code infusion (SPECTRA)

Cursor skills and `.cursor/rules` do **not** load here. These files do.

| Path | Loads when | What it is |
|---|---|---|
| `CLAUDE.md` (repo root) | every session | Two-agent split, never-list |
| `.claude/rules/spectra-stance.md` | every session | Thesis voice / Gilad frame |
| `.claude/rules/spectra-two-agents.md` | every session | Ops vs science, mail, git |
| `.claude/rules/spectra-ledger.md` | paper/queue files | TEST / keep-last / honest |
| `.claude/skills/spectra-thesis-mission/` | auto, or `/spectra-thesis-mission` | Identity, NEON, Drive IDs, state of mind |
| `.claude/skills/spectra-start/` | only `/spectra-start` | Session boot checklist |
| `docs/AGENT_MAIL.md` | when you read it | Thin ping bus |
| `docs/CLAUDE_RESEARCH_HANDOFF.md` | when `/spectra-start` says so | Full scientific handoff |

Do not `@`-import the handoff or the ledger into `CLAUDE.md`. That would load thousands of lines every turn and burn Max.
