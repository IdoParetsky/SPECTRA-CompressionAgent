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

**Final fine-tune: G2 for every P row (10 Oct; §348).** G2 = cosine from lr 0.1 at batch 256, CUDA-graphed, GPU crop + flip, 100 epochs, keep the last epoch (`SPECTRA_EVAL_FINAL_FT_SELECT=last`). VG2 found it SPEED-EQUIVALENT to the plain recipe on VGG-19 C100 (§348), which met Ido's 9 Oct 08:36 condition ("adopt_after_vgg"); §342's NR rows are the paper's narrow rows. Quote raw and honest. A table moves to G2 only through G2 runs: never mix G2 and plain rows, or keep-last and restore-best rows, in one comparison. §330's per-family recipes stay the record for rows run before 10 Oct.

No ImageNet DRL train. C100 enters the train pool only once recoverable.
