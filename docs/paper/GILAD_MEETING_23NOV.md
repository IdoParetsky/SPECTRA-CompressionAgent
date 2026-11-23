# Meeting with Gilad Katz, Monday 23 Nov 2026, 19:00 IST

Ido Paretsky and Dr. Gilad Katz. Minutes written by Ido in Hebrew; English translation by Claude Code, approved by Ido on 23 Nov. Materials shown: the meeting itinerary, the Think Tank and the paper cards (claude.ai pages), and the side-by-side results table `docs/paper/tables/SPECTRA_results_summary_23NOV.csv`.

## Bottom line (Ido's summary)

Gilad's view is that Ido and Claude let go of the DRL agent too early, in despair. He wants us to press harder: improve the agent's pipeline, methodology and implementation, holding to the possibility that under the right configuration the DRL agent will outperform the threshold rule. He believes in the Transformer architecture.

He wants an honest comparison of the DRL agent and the threshold rule against the published methods' results, side by side.

He sent four documents, originally written for a new MSc student (`docs/Gilad 23 nov 2026 - 01_NAP2_v2_*.docx`: the NAP2-v2 project sheet, project plan, feature catalogue and implementation spec, and the feature-families summary). For now we are to study the approaches and ideas in them and try to implement, integrate and test them as improvements to the SPECTRA DRL agent. He has already implemented some of them; Ido asked him for his NAP2-v2 repository (23 Nov). Ido gave Claude Code a GO to develop the science and engineering toward a better-configured DRL agent.

## Minutes (translated)

1. **An initial idea worth considering later: an equilibrium point between accuracy and pruning ratio.** Instead of setting a hard target for the network, build an equilibrium formula: weight and rank each reduction in network size (for example 10 %), with a linear return in the reward, alongside a penalty of some amount. The question becomes how much the network can be reduced without harming it too much. Suitable formulas must of course be formulated, tested and refined.
2. **Backtracking:** the agent performs an action; if it does more harm than good, roll back (keep versions) and prevent it from performing the same action again.
3. **Strengthening the method through Gilad's project extending NAP2** (the attached documents).
4. **A document for Gilad, soon, on how the agent works:** the algorithm, methods and rationale; to understand the Transformer, the modelling and the design; to understand what works and what doesn't. Claude prepares it as a first step, with a link to the code repository, readable by anyone so that Gilad's Claude agent has no access problems (the repository is public). Gilad will feed it to his Claude agent, which already knows him, and which will rewrite it as he wants to read it, with the emphases he likes. In effect: a summary of the SPECTRA project and of the agent so far, without the internal jargon of Ido's Claude agent, as if it were an academic paper accessible to any expert in the field with no prior knowledge of the project.
5. **Suggestions for features that can be implemented, merged and integrated into the agent,** after Gilad reads the current pipeline, methodology and implementation.
6. **A summary of the project and of the agent so far.**

Translator's readings, approved: in item 1 the penalty is read as a penalty for accuracy lost; in item 4 "which already knows him" refers to Gilad's own Claude agent knowing Gilad's preferences.

## What follows (Claude Code, 23 Nov)

- The overview document (item 4) is drafted first: `docs/paper/SPECTRA_OVERVIEW_FOR_GILAD_23NOV.md`, reviewed by Claude Code before Ido sends it.
- A study of the four documents and the older NAPv2 code maps each idea to the agent (state features, reward or value estimate, plan selection), with cost, risk and the cheapest deciding test: `docs/paper/lit/GILAD_NAP2V2_DIGEST_23NOV.md`.
- The agent-improvement plan then covers the reward's fidelity to the post-recovery result (where NAP2-style early-signal prediction plugs in), the equilibrium reward of item 1, backtracking (item 2) and richer state features, each registered in `docs/SITTING_GPU_QUEUE.md` before any run and judged against the threshold rule.
- The sealed read stays held, so that it judges the improved agent.
- The ImageNet short-recovery study (queue row SHORT-IMN) continues; it measures how well a short run predicts the 90-epoch result.
