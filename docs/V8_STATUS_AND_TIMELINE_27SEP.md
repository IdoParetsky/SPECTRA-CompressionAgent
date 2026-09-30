# V8 — what of the 21 Sep designs is implemented, what is not, and when (Fable, 27 Sep 2026 05:40 IDT)

Ido's question: were `V6_REPRESENTATION_DESIGN.md`, `GILAD_BENCHMARK_SETUP_21SEP.md`, `V7_OVERHAUL_PROPOSAL.md`, `V7_TRAIN_CATALOG.md` implemented? Item by item, with the reason and a GPU-aware timeframe. QOS cap is **4**; the factored train `21536398` holds one slot until it ends (episode ~235 of 250 + patience; 1–3 days); the ten 27 Sep jobs (`PROMPT_OPS_V8_QUEUE.md`) hold the rest for the next ~24 h, then the Budget+STOP train holds one slot for 7 days.

## 1. `V6_REPRESENTATION_DESIGN.md`

| Item | State | Why / when |
|---|---|---|
| Counterfactual probe (`SPECTRA_EVAL_COUNTERFACTUAL`) | **done, run** | §123: `state_used` 38 % (r20) / 53 % (r56) → the encoder is read. Gate for the next row is **open**. |
| Group-as-token + relational attention bias (`SPECTRA_STATE_TOKENS=groups`) | **not implemented** | It was gated on (a) the probe saying content is read — now true — and (b) the newest actor still not beating the heuristics at matched size — also true (no freeze of the area / PPO-8 / factored trains is TESTed). Cost ~400 lines + tests, one sitting. **Timeframe:** implement in the next development sitting (28 Sep). Its train needs a 7-day slot: the first one frees when `21536398` ends (28–30 Sep) → `v8-grouptoken` train enqueued then, in-band × area × 10-net, one change vs the area train. |
| Shared actor/critic trunk | not implemented | second cell, after group tokens have a TEST. |
| Full feature refresh under recipe A | flag exists (`SPECTRA_REFRESH_ALL_FEATURES`) | rides on the group-token train, not its own GPU. |
| Probes VGG-13 + r56-w6 | done | in all v6/v7 profiles. |

## 2. `GILAD_BENCHMARK_SETUP_21SEP.md` (the protocol note)

| Item | State | Why / when |
|---|---|---|
| Protocol lock (a)(b)(c), three cells, two operating points | **done** | `CATALOG_L_TEST_PLAN.md` §5–6; Gilad note sent-ready. |
| Train ∩ test = ∅, VGG-16 out of train, unit tests | **done** | `tests/test_v5_catalog.py`. |
| Same-loop controls on the twins (mild, L1) | **done** | §124 / §125. VGG-19 C100 collapses under Adam 1e-3 for every method — L3 has no valid row until a recipe passes the C100 gate (§4 below). |
| DepGraph checkpoint loadability (their `.pth` objects) | **done for L1** (`21703457`, `21703461`): both files are plain state_dicts; the ResNet-56 one loads strict-clean into `resnet_chenyaofo.resnet56` and scores 93.43 % on 3000 test images (paper 93.53) → `configs/input_catalog_l_depgraph_r56.json`; same-loop mild / L1 controls on it **queued** (`21703466/67`). **L3 pending**: VGG-19 C100 uses DepGraph's own key layout (`block0.0 … block4.10`, one `classifier`) → ~40-line factory next sitting, and no valid row until a recipe recovers CIFAR-100. |
| Size-matched (2.57×) rows | not run | needs `SPECTRA_EVAL_MIN_FLOP_RATIO`/extra passes on the L cells — heuristic jobs, ~5 h each; run right after the DepGraph-checkpoint controls (same week). |
| Frozen-agent rows on L1–L3 | not run, by design | waits for an actor that meets bar 2 on the coverage cells (no current freeze does). |
| Budget table | not filled | slurm elapsed of one train + one TRAJ + their `reproduce` epoch counts × a measured CIFAR epoch — ops task, no GPU; can be written this week. |

## 3. `V7_OVERHAUL_PROPOSAL.md`

| § | Item | State | Why / when |
|---|---|---|---|
| 1.1 | `SPECTRA_PROBE_SCORE=area` | **done, used** by the area / PPO-8 / factored / budget trains | none of the area freezes is TESTed yet (Ido's GO). |
| 1.2 | PPO 8×8 sample reuse | **done, ran** (`21536397`, freeze ep143 area 0.0675) | untested; ops §0c: do not crown PPO-8 from the probe. |
| 1.2 | deterministic FT seed per step | not implemented | ~20 lines; a *variance* lever that does not change the recipe — candidate ride-along for the group-token train, or its own cheap no-agent A/A test (same walk twice) to measure the FT noise floor first (~1 h GPU). Next sitting. |
| 1.2 | incremental credit (`SPECTRA_REWARD_INCREMENTAL`) | not implemented | reward change = new actor; only after Budget+STOP is read. October. |
| 1.2 | Δacc surrogate from the ~25 k logged transitions | not implemented | **GPU-free** (CPU regression over `reward_trace.jsonl` + step records). Worth a CPU job this week as analysis; as a critic prior later. |
| 1.3 | `SPECTRA_FT_LR`, re-gate | done, ran | both constants fail the thin control (§117–§121). |
| 1.3 | BN recalibration | **done today** | control `21703437`; rides on A-LSQ / C-PCA / schedule arms. |
| 1.3 | one-recipe schedule (AdamW warm-up cosine; RAdam) | **done today, gate queued** | `21703438–41`. |
| 1.4 | in-band linear default | done | every new train. |
| 1.5 | counterfactual probe | done, run | §123. |
| 1.5 | standardizer OOD check on thin nets | not done | CPU, 30 min; next sitting. |
| 2.3 | A-LSQ, C-PCA | **done today, running** | `21703433–36`. |
| 2.3 | C-G-KD | not implemented | diagnosis only; skip unless Gilad asks. |
| 3.1–3.2 | Budget + STOP | **done today, train queued** | `21703443`. |
| 3.3–3.7 | pointer policy, batch-then-FT, hindsight τ, width-adaptive ladder, per-net normalisation | not implemented | after Budget+STOP and group tokens are read; each is a new actor (7-day slot). |
| 5 | audit checklist A1–A10 | A1 done (fixed), A4 done (BN recal), A6 not checked, A2 not checked | A2/A6 are CPU checks; next sitting. |

## 4. `V7_TRAIN_CATALOG.md`

| Item | State | Why / when |
|---|---|---|
| 16-net diverse file + gate table + candidates input | **done** | `configs/database_offline_v7_diverse*.json`, `v7_c100_gate.json`. |
| Constant-LR re-gate (Adam 1e-4, SGD 0.01) | **done, failed** | admits 4/8 and 2/8 but both lose the CIFAR-10 thin control (§117–§121). |
| Schedule gate (the replacement) | **queued today** | `21703438–41`; pass = thin within 0.5 pp of §120 **and** ≥ 4/8 admits. |
| V7 diverse train | **not started, gated** | starts only if a schedule arm passes both gates → emit with `--min-c100 4` → train `v7-inband-diverse-area` in the next free 7-day slot (needs a second slot after Budget+STOP; realistic start when `21536398` ends **or** after the group-token train, i.e. early October unless Ido re-prioritises). |
| SVHN out of the train catalog (both cheap datasets truly held out) | not done | ops §0.6 asks for it once A-LSQ has a `val_best`; a 9-net C10-only file already exists (`database_offline_v5_p5b3_c10core.json`). One-line profile change; next sitting. |
| Fallback used today | P5-B2 (10-net: 9 C10 + VGG-11 SVHN) | the live catalog of every 27 Sep train. |

## 5. Recommended GPU order after today's ten jobs (Ido decides at 12:00)

1. **Read A-LSQ / C-PCA** (thin lands within ~4 h; twins ~5 h). Decision: recipe for future heuristics and for the *next* train.
2. **Read the schedule pair** (~6–8 h). Decision: CIFAR-100 in or out; if in, V7 diverse train becomes the next 7-day job.
3. **Budget+STOP** starts on its own when a walk frees a slot (tonight). Its first freeze is TESTed only on GO.
4. When `21536398` ends: the freed slot goes to **group-as-token** (if implemented by then) or to the V7 diverse train (if the schedule passed) — Ido's call; my recommendation: group-as-token first (novelty gate is open), V7 diverse second.
5. Heuristic holes (≤ 5 h each, any time): DepGraph-checkpoint twin controls; size-matched (2.57×) rows on L1/L2; FT-noise A/A pair.
6. CPU, no slot needed: Δacc surrogate over the logged transitions; standardizer OOD check; budget table.

Not this cycle: C-G DRL, a second factored train, another ranking menu, BERT, ImageNet DRL, 200-epoch SGD inside training.

---

# 28 Sep 02:00 IDT — status after the first V8 night (Fable)

Cell labels **L1/L2/L3 are retired** (they collide with L1/L2 pruning). Use **R56·C10**, **VGG16·C10**, **VGG19·C100**. "Catalog L" remains the name of the hold-out catalog file.

## 6. What landed (ledger §126–§131)

| Cell | Result | Verdict |
|---|---|---|
| A-LSQ (§126) | thin r20 −4.1 @ 0.536 (A −3.4); thin r56 **−6.2 @ 0.923** (A −6.6); twin −4.3 @ 0.661 (A −3.3) | kinder on 1 of 3 → fails "≥ A on both"; stays an off switch |
| C-PCA (§127) | r20 −6.0 @ 0.536; r56 −7.7 @ 0.975; twin −5.2 @ 0.946 | worse everywhere → **crossed off**; layer replacement (random or generated) is closed for CNNs |
| BN-recal alone (§128) | r20 −4.1; r56 −6.9 @ 0.923 | no gain → internal caption only (Ido decision 4) |
| AdamW warm-up cosine (§129) | thin fails (−7.1 / −7.8 @ 0.832); C100 0/8 | crossed off |
| RAdam (§130) | thin fails (−5.6 @ 0.655 / −8.2 @ 0.832); C100 2/8 (VGG-11/13) | crossed off |
| DepGraph R56 anchor (§131) | mild **−3.1 @ 0.661/0.662**; L1 **−3.7 @ 0.575/0.482** (twin: −3.3 / −5.1 @ 0.415) | the τ-matched same-loop rows of R56·C10 on DepGraph's own weights |

Reading of the recipe question: every *rate* / *schedule* change that helps CIFAR-100 in 12 epochs hurts the CIFAR-10 control. The remaining uniform lever is the **budget** → cap-40 / patience-4 arms at 1e-3 (`21715233/34`) and 1e-4 (`21715235/36`) submitted 01:00.

## 7. Budget + STOP — the 27 Sep job was mis-profiled; re-submitted

`21703443` ran the generic default branch (tree_v8 lacked the `offline_train_v7_*` gate line that was patched locally after staging): rates {1.0, 0.9, 0.8}, old leap 10-net DB, 5-step episodes, no PPO. Cancelled 00:55 (Fable's own job), tree fixed, re-submitted as **`21715228`** — verified at 00:52: budget menu, `cbrt_cubes`, p5b2, area probe, STOP scale 100, PPO, per-step budget mapping lines, 1 STOP in 26 episodes. Lesson written into the ops prompt: confirm `compression_rates` in the Namespace line on every new profile's first poll.

## 8. Answers to Ido's 28 Sep questions

**V6 representation — what were we waiting for?** Two gates, both now closed: (a) the counterfactual probe had to say the encoder is read (it is: 38 % / 53 %, §123) — otherwise a richer state could not matter; (b) a first *clean-catalog* actor had to exist as the control, so the representation cell is one change against it (area train `21536396`). Nothing else was needed from the running trains. **Group-token and shared trunk are separated**: group-token is the cell (`SPECTRA_STATE_TOKENS=groups`; `src/group_tokens.py`; relation bias in `SpectraStateEncoder`; contract key; 6 new tests; full suite 304/304 on the cluster conda); the shared trunk is a later, separate cell so that a read stays attributable. **A/B stack:** in-band linear × P5-B2 × area probe × fpgm 5-action menu × recipe A × 12/4 — identical to `21536396` except the state. **Implemented and queued:** `v8-grouptoken` **`21716380`** (tree_v8b, nice 30), starts when a slot frees. Read on the diagnostic pair against `21536396`'s freeze at equal keep; the `token_feature_dim` in its `policy_config.json` must be 4 larger than the control's. Insight so far: none — no group-token weights exist yet; the only representation evidence is the probe (state is read) and the saturation diagnosis (the *selection* was the blind part, not the encoder).

**V7 overhaul — standing.** Implemented and tested: area score (used by every new train), PPO-8 (ran, untested freeze), `SPECTRA_FT_LR` + re-gates (failed), BN-recal (no gain), schedule arms (failed), A-LSQ / C-PCA (failed pass rules), Budget + STOP (training since 00:52), counterfactual probe (run). Not implemented: deterministic FT seed, incremental credit, Δacc surrogate, C-G-KD, pointer policy / batch-then-FT / hindsight τ / width ladder / per-net normalisation, standardizer OOD check, audit items A2/A6. Insight: of the seven "moving parts", two were real and are fixed (reward cube-root; saturated selection score), two are being tested by new actors (action semantics → Budget+STOP; state → group tokens), one is answered negative (recovery recipe variants do not move the walk), and per-step SNR remains the open one (no cell yet). **Next dev phase:** when Budget+STOP or group-token freezes and is walked (ops flags → Ido opens Fable), or when the cap-40 pair decides the catalog. Cells for that sitting: deterministic FT seed A/A (1 h GPU), Δacc surrogate (CPU), DepGraph VGG-19 loader, size-matched (2.57×) rows on R56·C10 / VGG16·C10, hold-out imports (ResNet-164, DenseNet-40 C10, PruningBench R18/R50 C100, Plain-20).

**Catalog.** The P5-B2 catalog is the *current* live set, not a design goal; the design is the 16-net diverse file, emitted from the gate. Wording "until then / temporary" removed from the docs; the ImageNet-cost paragraph is in the Gilad note §3.

## 9. Results feed (ops appends here — decision 2; this is Fable's entry point next sitting)

V8 RESULTS FEED — 28 Sep 17:48 IDT
C/D. cap-40 recipe: thin fail both LRs (§132–§133). C100 1e-3 still first net unpruned (9h48m, 1080). C100 1e-4: VGG-11 **−3.3 @ 0.819** admit; VGG-13 **−4.5 @ 0.819** admit; ResNets/MN-v2×0.5 not admitted. Cannot save either arm. Catalog emitted? **no**.
B.  Budget+STOP 21715228: **COMPLETED** 13:58, 250 ep, rewind 3/3, best area **0.0273** at ep84, last probe 0.0239, **no snapshot** (bar 0.05). TEST only on GO. Ledger §135.
E.  Group-token 21716380: **R** 3h49m `ise-6000-04`. `SPECTRA_STATE_TOKENS=groups`. `token_feature_dim` **63 vs 59 (+4)**. PPO-3, ep~13, no freeze yet. Flag took.
A.  Factored 21536398: COMPLETED 11:37. Freeze ep0167 / 0.0608. TEST on GO. Ledger §134.
Crossed off: cap-40 thin both LRs; A-LSQ; C-PCA; schedules; Budget freeze never written. Open Ido: Q4; GO TESTs of area / PPO-8 / factored; whether to TRAJ Budget latest_best.

V8 RESULTS FEED — 28 Sep 20:15 IDT
Ido GO option **A**. Area TRAJ **21725471** R 3090 pin ep0083; factored TRAJ **21725472** R 2080 Ti pin ep0167; both `eval_c10_thin_traj` 2-pass det 40/10. Quote val_best at equal keep. Match → drop head; win on r56-w4 → factored becomes control.
Group-token **21716380**: first PROBE 18:20 area **0.0555** (r56-w6 0.031 / r20-w10 0.080), freeze `ep0011`. Then PD Priority, SKU-pinned rtx_6000, last log ~19:59 ep16. Do not scancel. Do not TRAJ ep0011 without a new GO.
Cap-40 C100 **21715234 / 21715236 CANCELLED** 20:10 (dropped value; thin already failed). Maintenance 29 Sep drain; backups `/home/paretsky/spectra_pre_maint_28sep/`.
Fable sitting is **code today** (width-ladder, 0.95 menu, 0.70/probe audit, VGG-19, FT kill table). Do not wait for TRAJ. Screen honesty: thin TRAJ is 2 nets, not coverage.

V8 RESULTS FEED — 29 Sep 01:52 IDT
Area TRAJ **21725471 COMPLETED** 23:10. `[eval] TRAJ val_best`: r20 **−5.1 @ 0.536/0.655** val −6.39; r56 **−6.8 @ 0.923/0.769** val −8.94. Same keep as 2-pass mild §93; r56 is the 90 %-rule clone. Ledger §136. Factored **21725472** still R 5h42m 2080 (r20 **−3.7 @ 0.536** val −4.02; r56 not yet). Do not drop the head until r56. Group-token **21716380** remains JobHeldUser. Science sitting has **v9** jobs R/PD — ops does not scancel them. Heartbeat 17:48 loop ended; re-arm. Backups NFS+Windows still running. Cluster SSH up at 01:52 (29 Sep drain still ahead).

V8 RESULTS FEED — 29 Sep 10:05 IDT
Twins **GO** §142: VGG-19 C100 **−6.7 @ 0.657** val −6.44 (legacy was unpruned). Adopt P as walk protocol. P-thin §143 r56 **0.739** vs N0 **0.923**. N0 three-seed r56 all 0.923. P-N4 §144 first undo step 77 not 39. DepGraph VGG-19 P §145 **−7.9 @ 0.534** (quote-only). Pickle still kills `final_ft`. Live: only **21726340** R. GT held. Next sitting: pickle on tree_v9c + C100-pool GO + one-change train. Do not fill idle GPUs from ops.

V8 RESULTS FEED — 29 Sep 15:55 IDT
Ido: full QOS utilization; independent TESTs no second GO. Sitting owns `docs/SITTING_GPU_QUEUE.md` and sbatches. Cluster 15:43: QOS 4, 0 R, holds only. `root_19` until 18:00. Sitting doc §PASTE + §12 restamped.

