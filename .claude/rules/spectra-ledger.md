---
paths:
  - "docs/paper/**"
  - "docs/SITTING_GPU_QUEUE.md"
  - "docs/LEARNING_PROGRAM_OCT8.md"
  - "docs/CLAUDE_RESEARCH_HANDOFF.md"
---

# Paper numbers live in git

Record of record: `docs/paper/RESULTS_LEDGER.md`. Chat and canvases are not.

When a TEST lands: ledger first (job, net, Δacc, params, FLOPs, seed, LOCKED vs PRELIM), then the matching draft table only with a GO, then a mail stamp. Skip akamaster ResNet-32. `param_ratio` / `flops_ratio` = fraction kept. Do not mix C100 DRL train returns into paper tables.

**Final fine-tune (Ido 8 Oct; §330).** Keep last epoch (`SPECTRA_EVAL_FINAL_FT_SELECT=last`). Recipe per family on **val**, never TEST: cosine from 0.1 full-width (DepGraph R56, VGG-19); cosine from 0.01 narrow (thin r56-w4, MobileNetV2 ×0.5). Quote raw and honest. Never compare keep-last rows with old restore-best-train-loss rows.

VGG-19 C100 stays on the **plain** recipe until a powered check says otherwise (§343). G2 is not a single-family rule until Ido calls.

No ImageNet DRL train. C100 enters the train pool only once recoverable.
