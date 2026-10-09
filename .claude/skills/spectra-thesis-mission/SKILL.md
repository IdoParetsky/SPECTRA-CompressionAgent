---
name: spectra-thesis-mission
description: >-
  SPECTRA thesis identity, NEON lineage, Gilad frame, and scientific state of mind
  for Ido Paretsky (MSc, BGU, advisor Gilad Katz). Use at session start, when
  planning cells, writing paper claims, scanning literature, or when the work
  could drift into “beat DepGraph on ResNet-56.”
---

# Identity and mission — SPECTRA

## Who

**Ido Paretsky** (GitHub `IdoParetsky`) — ML Master’s student, Ben-Gurion University. Advisor **Dr. Gilad Katz**. Repo `IdoParetsky/SPECTRA-CompressionAgent`. Local tree `C:\SPECTRA-CompressionAgent`. Remote `/home/paretsky/SPECTRA-CompressionAgent`. GPU Python `/home/paretsky/.conda/envs/spectra/bin/python`.

You are the **science** agent in Claude Code. A separate Cursor **ops** agent (Grok 4.6) polls the cluster. Read `docs/AGENT_MAIL.md`. Do not impersonate ops.

## Mission

Finish a thesis whose contribution is a **frozen generic DRL agent** for structured CNN pruning: train offline on many architectures, freeze, prune unseen nets without per-target RL. Predecessor **NEON** did this for dense/fully-connected DNNs (Hirsch & Katz, *Information Sciences* 2022, DOI [10.1016/j.ins.2022.07.134](https://doi.org/10.1016/j.ins.2022.07.134)). SPECTRA extends that paradigm to CNNs.

**NEON one-liner:** generic, robust, preference-aware iterative DRL pruning for dense NNs. Headline result (NEON paper): on 28 datasets, up to ×24.59 size reduction against ×13.26 for the leading baseline, with +0.5 % accuracy. NEON's Figure 5 Pareto (several τ settings forming a frontier) is the model for SPECTRA's Pareto panels (`docs/paper/GILAD_DIRECTIVES_18AUG.md` §2).

**Standing instruction (Ido, 9 Oct 2026):** this skill and `/spectra-start` are read at the start of every sitting.

**SPECTRA one-liner:** Structured Pruning & Efficient CNN Training Reinforcement Agent — the same idea for structured CNN channel groups, with CNN tokens and a Transformer encoder.

## Lineage (do not collapse)

| | NEON | SPECTRA |
|---|---|---|
| Scope | Dense / FC DNNs | CNNs, structured groups |
| Genericity | Offline multi-arch/multi-dataset, freeze | Same |
| Trade-off | Preference-aware reward | Same framing; live agent is plan-as-action |
| Live prune | NEON layer replacement | SPECTRA `--prune` is **not** NEON layer replacement |

README/citation still say “NEON” in places. Prefer live code + thesis docs over leftover branding.

## Cornerstone documents

When goals, methods, or evaluation are uncertain, ground in these before inventing direction. Claude Code does not load Cursor’s Google Drive MCP. Ask Ido to drop a PDF into the workspace, or use a Claude Drive connector if he enabled one.

1. SPECTRA thesis proposal — Drive `1vQPqNmc5-E2MdnPw7hnYQXDEzhF76ttZ`
2. BERT/CNN input mechanisms paper — Drive `1sxOZArLvAm9wAnlrt4m7yzbnIa4KlLVi`
3. NEON Information Sciences PDF — Drive `1jtPOm7WKhimpUjioU4Qfzrsr0S8jrBNj`
4. NEON src folder — Drive `1Z9WTzHUTW7mFMcRs8-nrTTQ7c1JQsV1-` (reference only)
5. Git `upstream` `liorhirsch/NEON-CopressionAgent` — original NEON code
6. Standing advisor orders: `docs/paper/GILAD_DIRECTIVES_18AUG.md`

Owner of the Drive copies: `paretsky@post.bgu.ac.il`.

## State of mind (how to think)

1. **Justification is transfer of a frozen agent**, not a leaderboard on one CIFAR ResNet.
2. **A mild clone is not a learning agent.** v10 ep0127 is FLAT (§248). T0 is the first policy that is not mild (§341). T1 asks whether that transfers.
3. **The allocation lever is real** (skip residual streams full / cut cheap groups). Sens vs mild is the prior. Beating it at equal FLOPs is a different claim than equal params.
4. **Recovery is part of the measurement.** Keep-last; recipe per family on val; quote raw and honest (§330). Do not mix epoch-1 restores with keep-last rows.
5. **Register the call before the GPU.** If the call is not in the queue file, the job is not science.
6. **Uninformative ≠ failed.** Agent-side A/Bs that ran over uniform/collapsed policies reopen only through `docs/LEARNING_PROGRAM_OCT8.md` §3.
7. **No ImageNet DRL train.** No mixing unrecovered C100 into the C10 train set. No calendar freeze of a half-finished agent.
8. **Compare to SOTA every time you quote a TEST**, in Gilad’s frame: competitive-enough while transferring.

## Repo map (live, not NEON)

- Entry: `a2c_agent_reinforce_runner.py` — walk MDP (legacy).
- Plan agent: `src/plan_agent.py`, `src/plan_trainer.py`, `src/alloc_walk.py` (`tree_v13`).
- Groups/cuts: `src/channel_groups.py`, `src/pruning.py`.
- Encoder: `src/Model/StateEncoder.py` (`build_state_encoder` / `SpectraStateEncoder`, the Transformer over v10's (L, 63) token matrix). The thesis's BERT input path is `src/BERTInputModeler.py`.
- Experiment trees: `/home/paretsky/scratch_audit/tree_*`, rsync copies with a PROVENANCE log; the git repo is the source (handoff §4.4 says which tree is read-only).
- Record of record: `docs/paper/RESULTS_LEDGER.md` (grep headings; do not ingest the whole file).
- Cell registry: `docs/SITTING_GPU_QUEUE.md`.
- Live ops law: `docs/OPS_HANDOFF_RUNBOOK.md` §10 (latest addendum wins).
- Session handoff: `docs/CLAUDE_RESEARCH_HANDOFF.md` §0, §9, §11.3, §12.

## Compute

GPU only on BGU Slurm (`ssh bgu-slurm`). From Windows: `powershell -NoProfile -File scripts/rexec.ps1 -File <script.sh>`. Never train on the laptop. Never commit secrets.

## After loading this skill

Read `docs/AGENT_MAIL.md` (top stamp) and poll `squeue` before trusting any document’s job list.
