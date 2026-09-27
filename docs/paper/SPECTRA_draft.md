# SPECTRA — paper draft (NEON format)

Working title: **SPECTRA: Multi-Objective Structured Pruning of Convolutional Neural Networks Using Deep Reinforcement Learning**

Ido Paretsky · advisor Dr. Gilad Katz · Ben-Gurion University  
Predecessor: Hirsch & Katz, *Information Sciences* 2022 (NEON) [1].  
**Numbers live in [RESULTS_LEDGER.md](RESULTS_LEDGER.md).** Do not paste a job summary mean here.

Status of this file: **DRAFT**. **Training freeze night of 17 Sep 2026** (no new reward A/Bs after that). **Paper including TESTs 30 Sep 2026** — no 6 Oct coverage buffer. First new-actor TESTs are in (§68–70); remaining spectrum catalogs must finish before 30 Sep.  
Section order matches NEON [1]: Introduction → Related Work → Approach → Evaluation → Results → Discussion → Conclusions.  
**Last restamp: 16 Sep 2026 10:55 IDT (Fable: §3, §4.1–4.2, §7 rewritten to the v3 method of record; §5 numbers unchanged).** Standing advisor orders: [GILAD_DIRECTIVES_18AUG.md](GILAD_DIRECTIVES_18AUG.md). Paper skeleton: [PAPER_SKELETON.md](PAPER_SKELETON.md). Cubes matching-std traj **21235566** r20 §75. Log1p cubes **21233227 COMPLETED** §73 is not the z-score curve. Prefer **21233371** r20 §74 / Path 3 **21233223** r20 §72 (r56 in FT). Similar **21230664** MobileNet **−2.1**. Train freeze **night of 17 Sep**. Keep L1. Queue **6 R**.

### Durable snapshot — 18 Aug 2026 08:55 IDT (do not rely on chat)

Write new TEST here and in the ledger in the same turn. This block is the memory.

- **CIFAR-10 architecture transfer (C1–C5, C10 FLOP-floor):** LOCKED. Similar + unlike inside τ except skinny-deep ResNet-56 at 70% params (−15.9/−16.2/−17.2). FLOP floor 0.70 puts that net inside τ at ~91% params / 70% FLOPs (−8.9/−9.2/−9.6).
- **Claim C9 (frozen 10-net → CIFAR-100):** LOCKED mixed. VGG-16 BN −7.5/−7.8/−7.3 inside τ. ShuffleNet-v2×1 **−3.9 / −3.4 / −4.3** three-seed (s44 −4.3 @ 0.837/0.823). Thin ResNets and RepVGG-A0 miss. Argmax s42 catalog **COMPLETED**: r20 **−16.2** and r56 **−17.9** miss; VGG **−8.3 @ 0.769** and ShuffleNet **−6.6 @ 0.726** inside; RepVGG **−11.3 @ 0.685/0.556** miss (§58). Ledger §21.
- **Param floor 0.80 walk:** r56-w4 still cliffs on three seeds (−21.7 / −25.7 / −22.5 @ ~0.80p / 0.54f). Easy r20 inside τ. Ledger §24.
- **Digit-MNIST LeNet (held-out dataset):** three-seed TEST **gain** +2.8/+2.7/+2.9 pp. Toy net. Ledger §22.
- **SVHN r20-w8 (held-out width, dataset was in the 10-net train mix):** −2.0/−1.5/−2.0 inside τ. Not in-catalog r20-w16. Ledger §23.
- **24-net:** Dedicated similar **20201260 COMPLETED** 21:08. s42: r20 **−5.9**; r56-w10 **−12.6** miss; r44 **−4.5**; VGG-19 **−2.8**; MobileNet **−1.7 @ 0.668/0.651**; DenseNet-100 **−1.7 @ 0.835/0.831**. Skip r32. s44 **20204215 COMPLETED**: r56-w10 **−8.6 @ 0.616/0.538** inside τ (not size-matched); DenseNet-100 **−2.4 @ 0.798/0.813**. Skinny **20201263 COMPLETED**: r56-w4 **−25.0 @ 0.704/0.499** miss (worse than 10-net −15.9 @ 0.704/0.550); easy r20-w2 **−5.7 @ 0.600/0.748**. Unlike **20201265 COMPLETED**: RepVGG-A0 **−4.9 @ 0.671/0.545**; A1 **−3.9 @ 0.662/0.552**; ShuffleNet **no TEST** (grouping / finetune fail). 24-net is not the remaining lever. Overnight fill uses the frozen 10-net actors. Ranking L2/SVD s42 **COMPLETED**: r56-w4 **−24.1 / −25.6 @ 0.667/0.482**. L2 s44 **20353577 COMPLETED**: **−23.4 @ 0.685/0.490** — two-seed greedy cliff (not size-matched to L1 **−15.9 @ 0.704/0.550**). Greedy L2/SVD r56 **−26.6 / −25.8 @ 0.667/0.454**. DRL L2 three-seed **−24.1 / −21.2 / −23.4** (s43 smaller). Unlike FLOP-floor s42 **COMPLETED**: ShuffleNet×1 **−1.9 @ 0.809/0.826**; ×1.5 **−2.0 @ 0.821/0.799**; RepVGG-A0 **−3.9 @ 0.792/0.702**; A1 **−4.3 @ 0.888/0.735** (milder than default unlike, already inside τ). SVD ranking three-seed r56 **−25.6 / −19.8 / −23.7** unmatched (keep L1). C100 residual FLOP-floor s42 **COMPLETED**: r20 **−5.7 @ 0.813/0.709**; r56 **−4.7 @ 0.871/0.702** inside τ (milder than SGD-no-floor −12.2 / −9.1). s43 r20 **−7.0 @ 0.881/0.702** (PRELIM; r56 in flight).
- **FPGM ranking A/B (PRELIM, ledger §59):** thin argmax + FPGM **21040934 COMPLETED**. r20-w2 **−4.8 @ 0.600/0.741** same size as L1 **−4.4**. r56-w4 **−23.4 @ 0.667/0.465** vs L1 **−25.2** same size — 1.8 pp milder, still a cliff. **Do not retrain.** BN-scale **21040935 COMPLETED** r20 **−3.8**; r56 **−27.1 @ 0.667/0.465** — **worse** than L1 (§60). Keep L1.
- **PROVENANCE CORRECTION (4 Sep audit, ledger §54).** Two eval-path defects affect how every row below was produced. (a) The frozen policy was **sampled**, not argmaxed, and the encoder's dropout was live at TEST: the same actor on the same net reaches parameter counts up to **10.7 pp apart** between `eval_train` and `eval_test` (154 of 277 net×job pairs differ). Seed spreads of 1–3 pp are inside single-actor resampling noise and must not be read as seed effects. (b) Every **prefer-Δparams/ΔFLOPs** arm bypasses the agent — `action_preferring_param_per_flop` discards the actor's action — so those rows are a *deterministic heuristic*, not DRL, and their "three seeds" are byte-identical. Sampled control **20945567 COMPLETED**: r20 **−3.9 @ 0.600/0.748**, r56-w4 **−25.4 @ 0.667/0.482** vs locked **−15.9 @ 0.704/0.550**. Argmax **20945568 COMPLETED**: r20 **−4.4 @ 0.600/0.741**; r56-w4 **−25.2 @ 0.667/0.465** — **not identity**; hard net is the cliff (path 3 vs locked **−15.9 @ 0.704/0.550**). Similar-argmax **20945570 COMPLETED** DenseNet **−2.1 @ 0.801**. C100 argmax **20945572 COMPLETED** §58. Reward retrain **20945574 COMPLETED** (`neon`+`cbrt`, FLAGS det=1) — train, do not quote. Eval **20945576 COMPLETED identity** §63 (actor `job20945574`). Band train **20945744** PD QOS (prio 1). Unlike look-ahead s43 **20382197 R** took the 576 GPU.
- **FLOP-floor prefer-Δparams/ΔFLOPs (LOCKED three-seed):** r56-w4 **−8.9 / −8.0 / −8.1 @ 0.704/0.872**. r20 **−2.7 / −2.7 / −1.9 @ 0.600/0.886**. Same param/FLOP point, all inside τ. **Heuristic, not DRL, and one run not three (§54.2).**
- **ImageNet 20308031 COMPLETED, no TEST.** Earlier truncated-JPEG attempt failed fine-tune. Do not quote 82.8%.
- **ImageNet s42 20318168 TIMEOUT.** No TEST. 7-day retry **20715875** PD on QOS after parent **20412393 COMPLETED** (`rtx_4090`). No ImageNet DRL.
- **ImageNet s43 20360208 COMPLETED (PRELIM, ledger §41):** frozen 10-net actor, truncated-JPEG MobileNet-v2 TEST **−4.6 @ 0.823/0.729** (0.719→0.673) inside τ. s44 **20382192 COMPLETED** **−5.1 @ 0.772/0.652** (0.719→0.668) inside unmatched. Two-seed PRELIM. s42 TIMEOUT. Not a SOTA ImageNet fight. No ImageNet DRL train. afterany Wave M **20412394** ShuffleNet×1 TEST **−1.7 @ 0.801/0.716**. Do not quote 82.8% or train-loader.
- **C100 SGD (PRELIM three-seed):** r56-w15 **−9.1 / −9.4 / −10.8** — s44 misses at 0.571/0.360. r20 **−12.2 / −10.1 / −8.8** — only s44 inside. Sizes not matched. Do not overwrite C9. Ledger §25.
- **Unlike FLOP floor 0.70 (LOCKED three-seed, ledger §28):** ShuffleNet×1 **−1.9 / −1.8 / −1.5**; ×1.5 **−2.0 / −2.3 / −2.8**; RepVGG-A0 **−3.9 / −4.6 / −4.6**; A1 **−4.3 / −4.2 / −4.2**. Milder than default unlike (already inside τ).
- **Unlike FLOP+prefer (LOCKED three-seed, ledger §30):** ShuffleNet×1 **−1.5 / −1.1 / −1.5** @ 0.871/0.944; ×1.5 **−2.1 / −2.1 / −1.9** @ 0.879/0.950; RepVGG-A0 **−4.4 / −3.7 / −4.0** @ 0.715/0.756; A1 **−3.5 / −3.5 / −4.1** @ 0.705/0.753. Same size. Operating point; default unlike already inside τ. **Heuristic, not DRL, and one run not three (§54.2).**
- **Unlike FLOP+prefer look-ahead (PRELIM one seed, ledger §62):** s42 **20715879 COMPLETED**. ShuffleNet-v2×1 **−1.0 @ 0.871/0.944**; ×1.5 **−1.8 @ 0.879/0.950**; RepVGG-A0 **−3.0 @ 0.715/0.756**; A1 **−2.5 @ 0.705/0.753** — all inside, **same sizes as §30**. Heuristic, not DRL. Catalog COMPLETED. Quote structural keep. Do not lock.
- **Unlike FLOP-floor greedy s42 (PRELIM heuristic, ledger §64):** **20715872 COMPLETED** 1 d 20 h 22 m (ended 10 Sep 08:12). ShuffleNet-v2×1 **−2.0 @ 0.801/0.716**; ×1.5 **−2.4 @ 0.771/0.701** (quote structural keep); RepVGG-A0 **−6.7 @ 0.847/0.701**; A1 **−6.5 @ 0.857/0.702**. All four inside τ. **Same sizes as FLOP-floor look-ahead §47.** Heuristic, not DRL. Do not quote wrap job-mean. s43/s44 still HELD. Do not lock.
- **C100 unlike-extra argmax (PRELIM, ledger §67):** s42 **21168557 COMPLETED** 13.9 h. Frozen Path 3 actor, det=1. ShuffleNet-v2×1.5 **−5.9 @ 0.758/0.738** (0.742→0.683) inside τ — quote **structural** keep; do not quote masked 0.716. RepVGG-A1 **−12.7 @ 0.667/0.547** (0.764→0.637) miss. Extra unlike nets; do not overwrite §21 / C9. Do not quote wrap job-mean.
- **Prefer snapshot C10-thin argmax (PRELIM CLIFF, ledger §68):** **21229256 COMPLETED** 1 h 20 m (ended 13 Sep 07:26). Actor is continue **21184512** `latest_best` (prefer knob **off** at TEST, det=1). r20-w2 **−3.5 @ 0.600/0.753** (0.648→0.613) vs Path 3 **−4.4 @ 0.600/0.741**. r56-w4 **−23.8 @ 0.667/0.457** (0.888→0.650) vs Path 3 **−25.2 @ 0.667/0.465** — matched params, 1.4 pp kinder, still a cliff. **Not a WIN.** Dispatch labeled WIN (min-keep is r20) and queued spectrum + FLOP-0.70; skipped ep99. Prefer similar **21230664 R**. Do not quote wrap **−0.14 pp** or train 73.23. Do not overwrite locked −15.9.
- **Cubes snapshot C10-thin argmax (PRELIM, ledger §69):** **21229257 COMPLETED** 45 m (ended 13 Sep 08:11). Actor is continue **21184514** `latest_best` (prefer knob **off**, det=1). r20-w2 **−4.3 @ 0.600/0.741** (0.648→0.605) — size-matched **tie** vs Path 3 **−4.4 @ 0.600/0.741**. r56-w4 **−21.9 @ 0.722/0.524** (0.888→0.669) miss at a **larger** keep than Path 3 0.667 — unmatched, still fails τ. Do not quote wrap **−0.13 pp** or train 75.04. Spectrum catalogs **21230675–678** PD QOS.
- **Prefer snapshot similar-family (PRELIM in flight, ledger §70):** **21230664 R**. r20-w16 **−5.2 @ 0.669/0.614** vs Path 3 **−5.1 @ 0.669/0.649** — matched-params tie. r56-w10 **−22.7 @ 0.613/0.344** vs Path 3 **−12.8 @ 0.661/0.421** — unmatched, **deeper miss**. r44 **−8.1 @ 0.620/0.356** vs Path 3 **−4.5 @ 0.699/0.542** — inside τ, unmatched, **3.6 pp worse**. VGG-19 **−3.0 @ 0.679/0.706** vs Path 3 **−2.7 @ 0.699/0.738** — inside τ, unmatched. MobileNet **−2.1 @ 0.691/0.590** vs Path 3 **−2.4 @ 0.688/0.587** — near-matched **tie**. Skip r32. DenseNet in FT. Do not quote wrap.
- **Cubes C10-thin trajectory (PRELIM COMPLETED, log1p, ledger §73):** **21233227 COMPLETED** 2 h 31 m (ended 13:46). r20 val_best **−3.5 @ 0.698/0.760**. r56 val_best **−7.0 @ 0.965/0.792** (val **−9.82 pp**, inside τ; almost no compression). r56 terminal **−8.8 @ 0.894/0.678** has val **−11.65** over τ — do not pick it. `floor_cross` NONE. Not a cliff fix vs pad **−21.9 @ 0.722**. Do not quote wrap **−0.06 pp**. Matching-std redo is §75 — do not quote this log1p catalog as the cubes policy.
- **Cubes C10-thin trajectory r20 with matching std (PRELIM in flight, ledger §75):** **21235566 R**. Std loaded n=721. TRAJ floor_hold **−1.5 @ 0.824/0.809**; val_best **−8.6 @ 0.442/0.658** (val **−9.01 pp**); terminal **−17.0 @ 0.378/0.633** (val **−17.85**, over τ — do not pick). Same deep walk as Path 3 §72 **−8.1 @ 0.478**, not the shallow log1p §73. r56 in FT.
- **Prefer snapshot C10-thin trajectory (PRELIM identity, invalid protocol, ledger §71):** **21233226 COMPLETED** 4 m 12 s. TRAJ **+0.0 @ 1.000/1.000** on r20-w2 and r56-w4. T1105 freeze had no `standardizer.pt`; eval used **log1p fallback**. Not a prefer-policy TEST.
- **Prefer C10-thin trajectory r20 with matching std (PRELIM in flight, ledger §74):** **21233371 R**. Std loaded n=721. TRAJ floor_hold **−2.9 @ 0.763/0.785**; floor_cross **−3.6 @ 0.682/0.753**; val_best=terminal **−7.5 @ 0.470/0.670** (val **−7.88 pp**). Near-clone of Path 3 traj §72 (**−3.0 / −4.8 / −8.1**). Not identity. Do not mix with pad §68 **−3.5 @ 0.600**. r56 in FT.
- **Path 3 C10-thin trajectory r20 (PRELIM in flight, ledger §72):** **21233223 R**. Log1p. TRAJ floor_hold **−3.0 @ 0.755/0.772**; floor_cross **−4.8 @ 0.673/0.740**; val_best=terminal **−8.1 @ 0.478/0.663** (val **−9.04 pp**, inside τ). pass 1/1 params **x0.400** at the same Δacc — different counter; quote TRAJ. r56 in FT. Do not mix with pad **20945568 −4.4 @ 0.600/0.741**.
- **Chain B neon+cbrt thin eval (PRELIM identity, ledger §63):** **20945576 COMPLETED** 3 m 22 s. Actor `job20945574`, FLAGS det=1 neon. r20-w2 **+0.0 @ 1.000/1.000**; r56-w4 **+0.0 @ 1.000/1.000**. Frozen-actor Path 3 **unchanged** (20945568 still prunes). cbrt-only retrain did not yield a pruning argmax. Band **20945744** CANCELLED 10 Sep 04:15 (5-step train); replacement **21194543 R** (128-step, not TEST). Overnight C10 retrains continued past 36h (full-net, **not TEST**): prefer **21184512**, cubes **21184514**, floor **21184407**, F1 **21184409**. Do not overwrite locked −15.9 / C9.
- **CIFAR-100 DRL residual eval (PRELIM one seed, ledger §31):** **20353582 COMPLETED**. r20-w16 **−8.3 @ 0.673/0.627**; r56-w15 **−8.4 @ 0.662/0.469**. Both inside τ. Do not quote train returns. Do not overwrite frozen-C10-agent §21.
- **C100 residual SGD FLOP-floor (ledger §25):** r20 **−5.7 / −7.0 / −7.7** inside. r56 **−4.7 / −10.4 / −6.4** — s43 **misses at 95% params**. FLOP+prefer: r56 **−8.2 / −4.5 / −8.5 @ 0.703/0.872** three-seed inside; r20 s44 **−10.4** miss. Do not overwrite C9 Adam-40.
- **Similar-family FLOP floor (PRELIM, ledger §29):** three-seed COMPLETED. DenseNet **−2.2 / −2.4 / −2.2** @ 0.837/0.833, 0.822/0.827, 0.780/0.797. r56-w10 s43 **−14.3 @ 0.946/0.700** still a miss. s42 **−5.9** / s44 **−5.6** inside at ~90% params.
- **Similar-family FLOP+prefer (LOCKED three-seed including DenseNet, ledger §33):** s42/s43/s44 **COMPLETED**. r20 **−3.8 / −3.5 / −3.8 @ 0.717/0.880**. r56-w10 **−3.8 / −4.0 / −4.4 @ 0.702/0.872**. r44 **−2.3 / −2.6 / −2.6 @ 0.702/0.872**. VGG-19 **−2.2 / −2.6 / −2.4 @ 0.837/0.923**. MobileNet **−1.9 / −2.0 / −2.2 @ 0.767/0.912**. DenseNet **−2.0 / −1.9 / −2.7 @ 0.870/0.951** (s42 0.949→0.929). Prefer is the lever — but it is a **same-loop heuristic lever, not the agent**, and these are one run reported three times (§54.2).
- **Similar look-ahead greedy (PRELIM, ledger §34):** s42/s43/s44 **20382177 / 178 / 179 ALL COMPLETED.** r20 three-seed **−7.4 / −7.0 / −7.3 @ 0.713/0.525** inside. r56 three-seed **−22.4 / −20.1 / −22.2 @ 0.702/0.376** cliff (s42 0.959→0.735). r44 three-seed **−8.0 / −9.0 / −8.0 @ 0.703/0.391** inside τ. VGG-19 three-seed **−3.5 / −3.2 / −2.5 @ 0.703/0.669** (s42 0.934→0.899) inside. MobileNet three-seed **−3.2 / −3.3 / −3.0 @ 0.708/0.511** (s42 0.938→0.906) inside. DenseNet three-seed **−2.4 / −2.6 / −2.6 @ 0.701/0.679** (s42 0.949→0.925) inside. Catalogs COMPLETED. Do not lock.
- **C100 C9 Adam-40 random (PRELIM, ledger §40):** s42 **20412533 COMPLETED**. RepVGG-A0 **−13.4 @ 0.662/0.498** (0.753→0.619) miss unmatched; three-seed **−13.4 / −12.8 / −12.6**. Catalog COMPLETED PRELIM. r20 three-seed **−18.0 / −17.0 / −17.3** miss unmatched; r56 three-seed **−26.2 / −26.4 / −30.3** miss unmatched, worse than default **−17.8**; VGG three-seed **−9.3 / −8.5 / −8.5** inside unmatched; ShuffleNet three-seed **−4.2 / −5.5 / −4.4** inside unmatched (quote structural keep). Same family split as default. Designed leaves — do not attach Wave O. Do not overwrite §21.
- **C10-thin eval τ=5 (PRELIM three-seed COMPLETED, ledger §32):** s42 r20 **−4.4 @ 0.600/0.748**; r56 **−23.9 @ 0.667/0.481**. s43 r20 **−4.4 @ 0.600/0.780**; r56 **−20.1 @ 0.593/0.475**. s44 r20 **−2.5 @ 0.600/0.794**; r56 **−21.0 @ 0.685/0.490** (0.888→0.678). Easy net stays 60% params (s44 milder, inside τ=5). Skinny ResNet cliffs on all three seeds at unmatched sizes. Tightening eval τ did not make the frozen τ=10 actor milder on the hard net. Do not expand τ=5.
- **C100 C9 Adam-40 FLOP+prefer (LOCKED mixed, ledger §35):** s42/s43/s44 **20381800 / 801 / 802 COMPLETED**. Three-seed r20 **−14.5 / −13.8 / −12.7 @ 0.716/0.879** miss; three-seed r56 **−12.0 / −11.4 / −10.5 @ 0.703/0.872** miss; three-seed VGG **−7.6 / −7.7 / −7.3 @ 0.838/0.931** inside; three-seed ShuffleNet **−3.5 / −4.1 / −4.6 @ 0.873/0.944** inside; three-seed RepVGG **−13.0 / −12.5 / −13.0 @ 0.719/0.756** miss, same sizes. Prefer is not an Adam-40 residual/RepVGG rescue. **Heuristic, not DRL, and one run not three (§54.2).** Child FLOP-only **20382180 COMPLETED**. Do not overwrite §21.
- **C100 C9 Adam-40 FLOP-floor only (PRELIM, ledger §36):** s42/s43/s44 **20382180 / 181 / 182 COMPLETED**. r20 **−11.8 / −12.1 / −10.6** miss unmatched; r56 **−10.4 / −11.7 / −12.0** miss unmatched; VGG **−8.4 / −8.4 / −7.7** inside unmatched; ShuffleNet three-seed unmatched **−3.4 / −3.9 / −4.0** inside (s42 **@ 0.881/0.858**, quote structural keep); RepVGG **−10.2 @ 0.886/0.761** miss vs s43/s44 **−8.2 / −7.4** inside only at ~95% params. Prefer still misses RepVGG at 72%/76%. Do not overwrite §21.
- **C100 C9 Adam-40 look-ahead greedy (PRELIM, ledger §37):** s42/s43/s44 **20412388 / 389 / 390 COMPLETED**. r20 three-seed **−18.1 / −17.3 / −17.9 @ 0.716/0.525** miss; r56 three-seed **−34.6 / −32.1 / −33.2 @ 0.701/0.357** cliff; VGG three-seed **−8.8 / −8.6 / −9.0 @ 0.701/0.672** inside; ShuffleNet three-seed **−6.2 / −6.4 / −5.9 @ 0.736/0.682** inside (quote structural keep); RepVGG three-seed **−11.4 / −11.7 / −12.1 @ 0.709/0.577** miss. Catalogs COMPLETED. Not an Adam-40 residual/RepVGG rescue. Do not lock.
- **C100 C9 Adam-40 mild (PRELIM, ledger §38):** s42/s43/s44 **20412530 / 531 / 532 COMPLETED**. r20 three-seed **−17.0 / −16.0 / −16.1 @ 0.669/0.649** miss; r56 three-seed **−18.2 / −16.8 / −17.7 @ 0.689/0.499** miss (not a look-ahead cliff); VGG three-seed **−8.0 / −8.3 / −8.0 @ 0.811/0.822** inside; ShuffleNet three-seed **−4.0 / −4.4 / −4.3 @ 0.860/0.835** inside (quote structural 0.860); RepVGG three-seed **−11.8 / −11.5 / −11.8 @ 0.684/0.548** miss. Same family split as default. afterok random **20412533 COMPLETED** (designed leaf). Do not overwrite §21.
- **Similar FLOP-floor look-ahead (PRELIM, ledger §39):** s42 **20412391 COMPLETED**. s43 **20412392 COMPLETED**. s44 **20412393 COMPLETED** DenseNet **−2.8 @ 0.798/0.700** (0.949→0.921) three-seed **−2.6 / −2.1 / −2.8** same size inside. MobileNet three-seed **−3.3 / −3.1 / −3.2 @ 0.933/0.700** (s44 0.938→0.906); VGG three-seed **−3.6 / −2.7 / −2.8 @ 0.804/0.701**; r44 three-seed **−4.2 / −4.4 / −4.4 @ 0.947/0.702**; r56 three-seed **−8.3 / −9.2 / −8.6 @ 0.952/0.702** inside (FLOP-floor-only s43 was **−14.3 miss**); r20 three-seed **−4.6 / −5.9 / −5.2 @ 0.908/0.701**. r32 skipped. Catalogs COMPLETED PRELIM. Floor stopped the unconstrained look-ahead r56 cliff on three seeds. ImageNet **20715875 CANCELLED** 10 Sep 02:13 (no TEST). Frozen ImageNet serial **21166873 R** (afterany fired early when 774 cancelled). Prefer-greedy **20715876 COMPLETED** DenseNet **−2.1 @ 0.870/0.951** (ledger §48). Do not lock.
- **Similar mild (PRELIM, ledger §42):** s44 **20382186 COMPLETED** DenseNet-100 **−2.3 @ 0.823/0.828** (0.949→0.926) two-seed **−2.1 / −2.3** same size inside; r20 **−6.8**; r56 **−13.9** miss; r44 **−4.2**; VGG **−3.5**; MobileNet **−2.3 @ 0.689/0.666**. s42 **20382184 COMPLETED** DenseNet **−2.4 @ 0.823/0.828** (0.949→0.925) three-seed **−2.4 / −2.1 / −2.3** same size inside; r20 **−5.9 @ 0.669/0.649** three-seed **−5.9 / −5.7 / −6.8** inside; r56 **−14.0 @ 0.661/0.421** (0.959→0.819) three-seed **−14.0 / −12.8 / −13.9** miss; r44 **−4.7 @ 0.699/0.542** (0.935→0.888) three-seed **−4.7 / −4.1 / −4.2** same size inside; VGG **−3.3 @ 0.811/0.819** (0.934→0.901) three-seed **−3.3 / −2.6 / −3.5** same size inside; MobileNet **−2.3 @ 0.689/0.666** (0.938→0.915) three-seed **−2.3 / −2.1 / −2.3** same size inside. Floor did not bind. s43 **20382185** DenseNet **−2.1** inside. Tracks DRL default s44 on r56; far milder than look-ahead cliff **−22.2**. Mild is not a τ rescue on the hard similar net. s42/s44 catalogs COMPLETED PRELIM. afterok **20382187** PD on QOS. r32 skipped. Do not lock.
- **Similar FLOP-floor mild (PRELIM, ledger §43):** s42 **20412538 COMPLETED**. r20 **−5.6 @ 0.801/0.706** inside; r56 **−11.6 @ 0.946/0.701** **miss τ**; r44 **−3.9 @ 0.905/0.703** inside; VGG-19 **−2.8 @ 0.811/0.819** inside (floor did **not** bind); MobileNet **−2.4 @ 0.791/0.703** inside (floor **did** bind vs unconstrained **−2.1 @ 0.689/0.666**); DenseNet **−2.2 @ 0.823/0.828** (0.949→0.927) inside (floor did **not** bind vs unconstrained mild **−2.1 / −2.3** same size). s43 **20412540 COMPLETED** catalog PRELIM. r20 **−5.1 @ 0.801/0.706** two-seed inside same size; r56 **−9.9 @ 0.946/0.701** (0.959→0.860) **inside τ** at the same size as s42 miss (two-seed split — do not lock); r44 **−3.6 @ 0.905/0.703**; VGG **−3.0 @ 0.811/0.819**; MobileNet **−2.3 @ 0.791/0.703**; DenseNet **−2.1 @ 0.823/0.828** (0.949→0.928) two-seed **−2.2 / −2.1** same size inside (floor did **not** bind). Skip r32. Child spoof **20884670 COMPLETED** (ledger §55). Do not lock.
- **Unlike look-ahead greedy (PRELIM, ledger §44):** s42/s43/s44 **20382196 / 197 / 198 ALL COMPLETED.** ShuffleNet-v2×1 three-seed **−2.2 / −1.9 / −1.8 @ 0.723/0.682**; ×1.5 three-seed **−2.5 / −2.4 / −2.4 @ 0.710/0.675** (quote structural; skip masked 0.666); RepVGG-A0 three-seed **−7.2 / −7.2 / −6.4 @ 0.709/0.577**; A1 three-seed **−6.3 / −6.1 / −6.3 @ 0.710/0.574** same size. Heuristic. Default unlike already inside; look-ahead is worse Δacc on RepVGG at similar size. Do not lock.
- **Unlike mild (PRELIM, ledger §45):** s42 **20412380 COMPLETED**. ShuffleNet-v2×1 **−1.6 @ 0.857/0.835** (0.924→0.908) inside (quote structural; do not quote masked 0.831); RepVGG-A0 **−5.3 @ 0.680/0.548** (0.943→0.890) inside; RepVGG-A1 **−4.2 @ 0.663/0.539** (0.944→0.902) inside; ShuffleNet-v2×1.5 **−2.6 @ 0.849/0.828** (0.932→0.906) inside (quote structural; do not quote masked 0.818). All four inside τ one seed. Look-ahead is worse Δacc on both RepVGGs. Child unlike-random **20412385 COMPLETED** (ledger §49). Do not lock.
- **Unlike random (PRELIM, ledger §49):** s42 **20412385 COMPLETED**. ShuffleNet-v2×1 **−1.6 @ 0.801/0.753** (0.924→0.908) inside (quote structural; do not quote masked 0.764); RepVGG-A0 **−6.2 @ 0.659/0.503** (0.943→0.881) inside; RepVGG-A1 **−5.3 @ 0.649/0.508** (0.944→0.891) inside; ShuffleNet-v2×1.5 **−2.2 @ 0.794/0.761** (0.932→0.910) inside (quote structural; do not quote masked 0.755). All four inside τ one seed. ×1.5 is milder Δacc than mild **−2.6 @ 0.849/0.828** and look-ahead **−2.5 @ 0.710/0.675** at unmatched sizes; default unlike already **−2.4 @ 0.818/0.801**. A1 is 0.6 pp worse than default unlike **−4.7 @ 0.650/0.521** at nearly the same size. Not a new transfer win. Catalog COMPLETED PRELIM. Child Wave Q **20412555 COMPLETED** §66. Do not lock.
- **Similar random (PRELIM, ledger §46):** First-pass keep: s43 **20382188** r20 **−6.7 @ 0.691/0.606**; r56 **−14.9 @ 0.628/0.368** miss; r44 **−6.2 @ 0.589/0.437**; VGG-19 BN **−3.1 @ 0.755/0.744**. s44 **20382189** first-pass r20 **−7.4**; r56 **−16.6** miss; r44 **−7.8**; VGG **−3.1 @ 0.741/0.719**. Restart (`Restarts=2`) s43 catalog **COMPLETED**: r20 **−6.9**; r56 **−17.6** miss; r44 **−6.9**; VGG **−2.6**; MobileNet **−2.4 @ 0.692/0.581**; DenseNet **−2.3 @ 0.735/0.754**. s44 restart catalog **COMPLETED**: MobileNet **−2.9 @ 0.694/0.615**; DenseNet **−2.5 @ 0.740/0.745** (0.949→0.924). Do not replace first-pass rows. r32 skip. s42 **20382187 COMPLETED** first-pass catalog PRELIM: r20 **−6.4 @ 0.688/0.588**; r56 **−18.2 @ 0.577/0.347** miss; r44 **−6.2 @ 0.590/0.415**; VGG **−3.5 @ 0.756/0.754**; MobileNet **−2.7 @ 0.697/0.554**; DenseNet **−2.3 @ 0.730/0.755**. Child unlike look-ahead **20382198 COMPLETED** (ledger §44). Child C100 recoverable **20884673 COMPLETED** — do not quote train. Random is not a look-ahead cliff and not a τ rescue on the hard similar net. Do not lock.
- **Unlike FLOP-floor look-ahead (PRELIM, ledger §47 / §65):** s42/s43/s44 **20412394 / 395 / 396 ALL COMPLETED**. Three-seed **same size**: ShuffleNet-v2×1 **−1.7 / −2.0 / −2.2 @ 0.801/0.716**; ×1.5 **−2.5 / −2.6 / −2.4 @ 0.771/0.701** (quote structural; skip masked 0.763 / 0.725); RepVGG-A0 **−6.9 / −6.6 / −7.1 @ 0.847/0.701**; A1 **−6.7 / −5.7 / −6.2 @ 0.857/0.702**. All four unlike nets inside τ. Heuristic, not DRL. Look-ahead is deterministic; seed spreads are FT noise. DRL FLOP-floor unlike is still better Δacc on both RepVGGs. Do not lock.
- **Similar FLOP+prefer greedy (PRELIM, ledger §48):** s42 **20715876 COMPLETED**. r20 **−3.9 @ 0.717/0.880**; r56 **−4.4 @ 0.702/0.872**; r44 **−2.6 @ 0.702/0.872** (0.935→0.909); VGG **−2.9 @ 0.837/0.923** (0.934→0.905); MobileNet **−2.0 @ 0.767/0.912** (0.938→0.918); DenseNet **−2.1 @ 0.870/0.951** (0.949→0.928) — all size-matched to DRL prefer. Prefer stopped the r56 greedy cliff. DRL keeps a small Δacc edge on VGG (−2.2 vs −2.9); DenseNet is a near-tie (−2.0 vs −2.1). Catalog COMPLETED one seed. Child **20715877 PD QOS**. Skip r32. Do not quote eval_train. Do not lock.
- **Similar FLOP+prefer look-ahead (PRELIM, ledger §66):** s42 **20412555 COMPLETED**. r20 **−3.4 @ 0.717/0.880**; r56 **−4.1 @ 0.702/0.872**; r44 **−2.6 @ 0.702/0.872**; VGG **−2.4 @ 0.837/0.923**; MobileNet **−1.9 @ 0.767/0.912**; DenseNet **−1.8 @ 0.870/0.951** — all inside, **same sizes as §33/§48**. Heuristic, not DRL. Look-ahead is a small Δacc edge vs prefer-greedy; DRL prefer still 0.3 pp better on r56/r44. Catalog COMPLETED one seed. Skip r32. Do not quote eval_train. Do not lock.
- **Similar FLOP-floor greedy (PRELIM, ledger §51):** s42/s43/s44 **20715868 / 870 / 871 ALL COMPLETED**. Three-seed **same size**: r20 **−4.8 / −5.7 / −5.1 @ 0.908/0.701**; r56-w10 **−9.2 / −9.3 / −9.0 @ 0.952/0.702** **inside τ**; r44 **−4.6 / −5.1 / −4.1 @ 0.947/0.702**; VGG-19 **−3.1 / −2.7 / −2.5 @ 0.804/0.701**; MobileNet **−3.1 / −3.0 / −2.8 @ 0.933/0.700**; DenseNet **−2.2 / −2.2 / −2.3 @ 0.798/0.700**. L1 greedy does not use the actor — spreads are fine-tune noise. Greedy still **ties** look-ahead once the FLOP floor binds. Skip r32. Do not lock.
- **C100 class-count spoof (PRELIM, ledger §55):** s42 **20884670 COMPLETED**. Same family split as locked §21: VGG-16 **−7.3 @ 0.893/0.926** inside; ShuffleNet-v2×1 **−4.2 @ 0.778/0.794** inside; r20 **−20.4** miss; r56-w15 **−18.2** miss; RepVGG-A0 **−11.5** miss. Spoofing the encoder's class-count token 100→10 did **not** rescue residuals. Diag **20930175 COMPLETED**: every non-identity C100 step was over-budget (empty band). Shaping **20967060 COMPLETED** (`structural_shaped` confirmed in FLAGS) — train, do not quote; kids=NONE. Matched-VGG DRL **20884671 COMPLETED** — train, do not quote. Residual eval **20884672 COMPLETED** (actor `job20884671`, sampled, no det=1): r20 **−12.0 @ 0.612/0.598 miss**; r56 **−10.4 @ 0.697/0.402 miss** §61. Do not lock.
- **C100 matched-VGG residual eval s42 (PRELIM, ledger §61):** **20884672 COMPLETED** 16 h 3 m (ended 9 Sep 04:35). Sampled, SGD-80. r20-w16 **−12.0 @ 0.612/0.598** (0.730→0.610) **miss** unmatched vs §31 **−8.3 @ 0.673/0.627** and milder than §56 s43 **−16.4 @ 0.597/0.611**. r56-w15 **−10.4 @ 0.697/0.402** (0.784→0.680) **miss** unmatched vs §31 **−8.4 @ 0.662/0.469** and §56 **−7.5 @ 0.623/0.495** (more params, fewer FLOPs). Matched-VGG train did not rescue held-out residuals. Do not overwrite C9 / §21 / §31. Do not quote `eval_train`.
- **Advisor 18 Aug:** Compare constantly to SOTA as a reference (do not claim home-court wins). Keep coverage **and** a NEON-style Pareto. No ImageNet DRL train; frozen C10 → ImageNet transfer now has two-seed TEST unmatched (ledger §41). **NAP2 scanned 20 Aug:** Michael’s NAPv2 is NAS performance prediction on NAS-Bench-201, not a pruner — complementary, do not lift. Minutes: [GILAD_DIRECTIVES_18AUG.md](GILAD_DIRECTIVES_18AUG.md).

Citation numbers **[1]–[78]** are those of the August 2024 thesis proposal, kept wherever the claim still holds. **[79]–[92]** are papers from the August 2026 literature survey (DepGraph was already [12] in the proposal). Tags: **LOCKED** / **DRAFT** / **TBD** / **CLAIM** — see [README.md](README.md).

**What this draft does *not* inherit from the proposal.** ImageNet [71] and Places365 [75] are not in the DRL fine-tune loop. Default keep-rates are `{1.0, 0.9, 0.8}`, not `{0.7, 0.6}`. Fine-tune is full-net, not layer-only. The reward is NEON’s preference function with *realized* param/FLOP credit, not the proposal’s skip-connection / filter-count cube. Frozen BERT is an ablation, not the default encoder. Channel grouping is environment bookkeeping, not a novelty claim (that layer is occupied by DepGraph [12] and SPA [81]).

---

## Abstract — DRAFT

Convolutional networks remain the workhorse of real-time vision, but their computational cost limits deployment where connectivity and compute are scarce [2, 5]. Pruning reduces that cost [6, 10]; *structured* pruning (channels, filters, feature maps) actually shrinks GPU work, unlike unstructured weight zeros [7, 11, 13]. Most structured methods, including recent architecture-agnostic grouping engines [12, 81], still *solve one target network at a time*.

SPECTRA (Structured Pruning & Efficient CNN Training Reinforcement Agent) extends NEON [1] from dense DNNs to CNNs: one offline actor-critic, trained on a catalog of architectures and datasets, then applied to unseen checkpoints without retraining the agent. A user-set accuracy budget τ (NEON’s preference-aware reward) trades size against accuracy. Unlike NEON, the environment *rebuilds* Conv2d/BatchNorm groups (residuals [3], DenseNet concat [4], depthwise ties) so reported parameter and FLOP ratios are real shape changes.

On CIFAR-10 [73], the 10-net agent keeps similar-family and unlike-family held-out networks inside a 10 pp budget, except skinny-deep ResNets. Learned schedules tie greedy filter ranking [18] on easy nets and beat size-matched greedy on the hard similar ResNet-56. Encoder capacity (BERT / wider / set) did not move that hard net; catalog diversity 3→10 did, and 24-net reversed it. The 10-net freeze is the scientific product of this thesis; a large shelf-product catalog is future work and needs a fresh hold-out. CIFAR-100 is a recoverability problem: VGG-11 BN [76] admits a ~10% structured cut with a test-set *gain* under a long SGD recipe; residual, DenseNet, and MobileNet families have not yet shown a comparable *no-agent* cut. The same frozen 10-net agent, applied to CIFAR-100 with no extra agent training, keeps VGG-16 BN inside τ=10 (−7.5 / −7.8 / −7.3 pp) and ShuffleNet-v2×1 (−3.9 / −3.4 / −4.3 pp, mask fallback), and misses on thin ResNets and RepVGG-A0 under Adam-40; an 80-ep SGD recipe (one seed) puts thin ResNet-56 w15 inside τ at a smaller size while ResNet-20 w16 still misses. Digit-MNIST LeNet, a held-out dataset, is a three-seed TEST gain (+2.8 / +2.7 / +2.9 pp). SVHN ResNet-20 width 8 (new width; SVHN was already in the train mix) stays inside τ (−2.0 / −1.5 / −2.0 pp). **[C1–C7 LOCKED; C8 LOCKED miss; C9 LOCKED mixed; MNIST + SVHN-w8 three-seed LOCKED; C100 SGD A/B PRELIM]**

---

## 1. Introduction — DRAFT

Convolutional Neural Networks (CNNs) have transformed computer vision — classification, detection, segmentation [2] — by stacking learned filters. Residual Networks [3] and DenseNets [4] made very deep stacks trainable via skip / dense connectivity. The cost of that accuracy is compute and memory. For edge and real-time settings the relevant question is not a leaderboard delta but a *user-set* accuracy budget: how much size can we drop without crossing τ.

CNN pruning [6, 10] removes redundant components. Magnitude, channel, and filter pruning [7, 11, 12, 18] work well on the network they were tuned for, and usually need another search when the architecture or dataset changes. Unstructured zeros [13–16] look strong on paper and do not reduce commodity-GPU inference cost. Structured cuts do, but residual adds and grouped convolutions make “keep 80% of filters” a different question from “keep 80% of parameters.”

NEON [1] showed that one preference-aware DRL agent, trained offline on many *dense* networks, can prune unseen dense nets without per-network retraining, and that a user can state the prune–accuracy trade-off in the reward. Applying that dense agent to *flattened* image data was the proposal’s negative control: modest size cuts and poor accuracy, worst on CIFAR-100 [73] (proposal §4). SPECTRA is the CNN-native answer to that gap — not a claim that NEON failed, a claim that image CNNs need a structured environment, a CNN-aware state, and a recoverability check before a dataset joins the train catalog.

**Contributions (LOCKED as intent; numbers in §5 / ledger):**

1. A structured CNN environment whose compression credit matches rebuilt layer shapes (residuals [3], DenseNet concat [4], grouped / depthwise ties). Grouping is infrastructure in the sense of DepGraph [12]; the thesis claim is the *policy*, not a new dependency algorithm.
2. A small Transformer state encoder [35] trained with A2C (one token per layer, coupling-aware attention). Frozen BERT remains an ablation; encoder capacity did not fix skinny-deep ResNets (ledger §16).
3. NEON’s preference-aware reward [1], with τ = 10 pp, scored on *realized* param/FLOP ratios. Same-loop greedy / mild / random rate-pickers (filter ranking after Li et al. [18]) so “beats greedy” is a schedule comparison, not a different importance criterion.
4. An offline train / similar / unlike protocol. On CIFAR-10, family and width transfer hold except skinny-deep ResNets. CIFAR-100 is limited by fine-tuning recoverability, not by “C100 missing from the 10-net train set.” The **10-net catalog is the scientific product of this thesis** (held-out families and datasets are the evidence). A large “shelf product” catalog — tens to hundreds of nets, trained so a deployed agent is ready for many arch × dataset pairs — is **future work** and needs a *fresh* held-out set; it is not a missing table here. 3-net → 10-net helped; 10-net → 24-net *hurt* the hard ResNet (ledger §17). Admit extra train nets on recoverability-band health, not on count.

NEON’s 28 tabular datasets are not re-run. ImageNet overnight fine-tune is out of the loop (**CLAIM**, freeze 15 Sep). ViT / DeiT are out of scope [42].

---

## 2. Related Work — DRAFT

This section follows NEON’s related-work order [1]: reinforcement learning, neural-network pruning, then neural architecture search. CNN background and structured-CNN methods sit inside §2.2. The arena is generic *offline* DRL for structured CNNs with a user τ, versus same-loop rate-pickers. It is not ImageNet ResNet-50 versus 2024 grouping engines.

Two kinds of generality are both real and sit on different layers. **Mechanism-level generality** — one *tool* that can prune many architectures — is occupied well by DepGraph [12] and SPA [81]. **Policy-level generality** — one *agent* that transfers across nets and datasets under τ — is NEON’s idea [1], moved to CNNs. No 2023–2026 paper in the survey closes that lane.

### 2.1 Reinforcement learning

“Reinforcement Learning (RL) is the problem faced by an agent that learns behavior through trial-and-error interactions with a dynamic environment” [46]. Applications range from robotics [47] and games [48] to routing [49] and dialogue [50, 51]. At step *t* the agent observes state *s_t*, picks *a_t* ∈ *A*, the environment returns *s_{t+1}* and a scalar reward [1].

Deep RL combines RL with deep networks [52]. SPECTRA, like NEON, uses an on-policy policy-gradient family. NEON’s derivation is REINFORCE [53]; the implementation is A2C. The policy update (proposal Eq. 1) is

*h_{t+1} = h_t + α · G_t · ∇ log p(a_t | s_t; h_t)*,

with *G_t* the return from *t*. AMC [79] is the canonical *CNN* RL compressor: a DDPG controller searches layer-wise compression for a *given* MobileNet / VGG. SPECTRA’s contrast is the same as NEON vs per-task RL: the actor is trained once on a catalog and frozen at eval. Lookahead-search RL for channel pruning [57] is likewise a per-target search, not an offline multi-net agent.

### 2.2 Neural network pruning

#### 2.2.1 From dense nets to CNNs

Neural-network pruning dates to the early 1990s [6]. Optimal Brain Damage [8] and Optimal Brain Surgeon [9] used second-order criteria at high Hessian cost [10]. Modern CNNs spend most inference time in convolutions, so removing whole feature maps / filters is the practical lever [7, 10, 18].

ResNets [3] add identity skips; DenseNets [4] concatenate all preceding feature maps. Those links couple channels: pruning a filter in one layer forces aligned cuts in add / concat partners. That is why a CNN pruner cannot treat layers as independent dense maps the way NEON could.

#### 2.2.2 Structured vs unstructured

Structured pruning operates at channels, filters, or feature maps and reduces the matrices the GPU actually multiplies [7, 11, 12]. Unstructured pruning zeros individual weights [13–15] and typically wins compression *ratio* [16] without winning inference on dense kernels. SPECTRA is structured only.

#### 2.2.3 Global vs local; generalizability

Benchmarking, from the proposal, still has two axes. **Global optimality:** does the method see the whole net [17–26], or only one or two successive layers [8, 27]? **Generalizability:** can it prune a previously unseen architecture and dataset without another controller training? NEON [1] is the dense-DNN existence proof of the second axis, with a user trade-off later echoed in OCNNA [28] and in interactive plans such as CNNPruner [29]. Filter pruning for efficient ConvNets [18] is the importance ranking SPECTRA *uses inside each layer*; the DRL policy only chooses the *rate*. Auto-balanced filter pruning [19], GDP [20], lottery tickets [21], importance estimation [22], Gate Decorator [23], layer-adaptive magnitude [24], manifold-regularized pruning [25], and ThiNet-style algorithms [26] are global or near-global *per-model* methods. Lookahead magnitude pruning [27] is a far-sighted alternative of the same ranking family — conceptually related to SPECTRA’s look-ahead *greedy* control, not to the learned actor.

Proposal Table 1 still has the right axes. CONVNETS [18], AFP [19], GDP [20], DeepPruningES [30], FPAC [37], multi-layer residual compression [34], and DepGraph [12] are automatic and often global; none combine offline multi-architecture / multi-dataset training with an explicit user τ the way NEON / SPECTRA do. DepGraph’s and SPA’s [81] “adaptability” is *grouping* adaptability, not policy transfer. AMC [79] has a resource budget on a *given* net, not a transferred τ.

**Table 1 — feature comparison (proposal Table 1, plus 2018–2024 neighbors).** Y = yes; N = no; P = partial. “Adaptability” for DepGraph / SPA = can group many architectures, not “one frozen agent on an unseen dataset.”

| Method | Non-greedy | Global view | Adaptability | Automatic | Comp.–acc. trade-off |
|---|---|---|---|---|---|
| CONVNETS [18] | Y | Y | Y | N | N |
| AFP [19] | Y | Y | Y | Y | N |
| GDP [20] | N | Y | Y | Y | N |
| DeepPruningES [30] | Y | Y | Y | Y | P |
| FPAC [37] | N | N | Y | Y | N |
| Multi-layer ResNet compression [34] | Y | Y | N | Y | N |
| DepGraph [12] | Y | Y | Y (groups) | Y | N |
| SPA [81] | Y | Y | Y (groups) | Y | N |
| AMC [79] | Y | Y | N (per target) | Y | P (resource) |
| MetaPruning [80] | Y | Y | P (family) | Y | N |
| SPECTRA | Y | Y | Y (policy) | Y | Y (user τ) |

#### 2.2.4 Mechanism-level generality: DepGraph and SPA

DepGraph [12] (CVPR 2023) models layer I/O dependencies and prunes coupled groups with a sparse-training + norm criterion, on CNNs, RNNs, GNNs, and Transformers. SPECTRA should thank it, then differentiate: skip/concat coupling and structured cuts are table stakes; they are not the thesis contribution. DepGraph *solves a given model*. It does not train an offline agent, does not encode τ as a reward, and does not transfer a schedule to an unseen net without re-solving.

SPA [81] (2024) attacks three practical barriers: coupled channels that differ by architecture, pruning at different training stages, and tools locked to one framework. It uses ONNX graphs and group-level importance (OBSPA is a post-training, calibration-light variant) and positions itself beyond DepGraph / OTO-v2. SPA’s “any architecture / any framework / any time” is about the *tool*. SPECTRA’s “any architecture / any dataset” is about a *transferred policy*. Using the same slogan would invite a fair referee objection. “Any time” in SPA means training stage, not “any previously unseen dataset without re-solving.”

HESSO [83] (OTO-lineage sparse optimizer) and Auto-Train-Once [82] (controller-guided prune-from-scratch, CVPR 2024) make train+prune less manual for a *given* DNN. FreePrune [84] automates across pruning granularities with training-free scores. All three are pipeline / criterion generality, not a cross-dataset DRL policy.

#### 2.2.5 2025–2026: still per-model

CNN structured pruning did not cool off. Metaheuristic channel search [85], comparative encodings for search-based CNN prune [86], learnable per-filter masks [87], differentiable attention-guided channel pruning [88], SVD-driven filter importance [89], flow-guided multi-architecture scores [90], and DepGraph-style coupling plus spectral entropy [91] all improve a criterion, a search, or a deployment pipeline for a given network. That raises the bar for “yet another pruning score.” It does not close the NEON→CNN offline-agent lane. Adjacent 2025–26 work — DualPrune [92], one-cycle structured prune [93], GCN search for channel rates [94], ℓ2,p structured sparsity [95], structured lasso + IB [96], knowledge-distillation plus structured lightweight CNNs, agentic prune/quant pipelines — sits in the same per-model bucket. RL papers with “pruning” in the title in this window are mostly domain RL, not a generic CNN offline agent; the closest DRL neighbor remains AMC [79]. Quote those papers as reference points on a Pareto plot. Do not claim SPECTRA beats them on their home architecture × dataset.

Other proposal-era CNN-pruning mechanisms still worth a clause: evolution strategy [30, 31], clustering / swarm [32], auto graph encoder-decoder [33], Transformer-related pruning under pretrain–finetune [36], FPAC [37], DETR pruning [38], global channel attention [39]. ConvNeXt [40], LLM pruning [41], ViT [42–44], and VAN [45] are out of SPECTRA’s structured-CNN loop.

MetaPruning [80] learns a PruningNet for channel configs of a target family — automatic, not an offline cross-dataset agent.

### 2.3 Neural architecture search

NAS explores automatic architecture design: search space, optimization method, candidate evaluation [1]. RL was applied to NAS in [54] and followed by [55–57]; evolution [58, 59], SMBO [60], and gradient-based search [61, 62] followed. Some works use CNN pruning to help NAS [63–65]; others use NAS to help CNN pruning [66–68]; a few prune or search at initialization without data [69, 70]. SPECTRA is **not** NAS. The agent does not invent a new topology; it walks an existing CNN and picks keep-rates. Keep this subsection short, as NEON did.

---

## 3. Approach — method of record (v3, 16 Sep 2026)

NEON §3 order: Overview, State, Action, Reward, Architecture, Training, Complexity. This section
describes the implementation that produces the paper's DRL rows (the "v3" agent; profiles
`offline_train_v3_*`). Earlier agents (the frozen 10-net actors and the v2 arms) are kept in
`docs/AUDIT_13SEP_OVERHAUL.md` as provenance; their rows in §5 are captioned there and are not
rewritten here.

### 3.1 Overview

SPECTRA is an offline, preference-aware DRL pruner in the NEON lineage [1]: one agent is trained
once on a catalog of pretrained CNNs and then applied, **frozen**, to architectures and datasets it
has never seen, without per-target training or adaptation. A pruning episode is a walk over the
network's prunable rows (every `Conv2d` / `Linear` that opens a channel dimension; the classifier's
output dimension is never a target). At each row the agent chooses one action from a small menu of
(keep-rate, filter-ranking) pairs; the environment resizes the whole **channel group** that row
belongs to, fine-tunes the network, measures validation accuracy and size, and returns NEON's
trichotomy reward. A walk makes **two passes** over the rows; a coupled group may be structurally
cut at most once per pass.

What distinguishes SPECTRA from a per-model pruner is that nothing in the loop is fitted to the
target network: the ranking criteria, the group rebuild, the fine-tune recipe and the accuracy
band τ are identical for the DRL agent and for every heuristic it is compared with (§4.1). The agent
contributes the *schedule*: which groups to cut, by how much, with which criterion, and when to
stop.

**Channel groups.** A `torch.fx` trace recovers the layers that must share a channel dimension —
producers joined by a residual add, segments of a concatenation, depthwise ties, and the norms and
consumers that read them (DepGraph-class bookkeeping [12], SPA-style criterion wrapping [81]).
Pruning a row rebuilds every producer of its group, the normalisations over that dimension and each
consumer's input slice in one consistent edit, verified by a dummy forward; a group that cannot be
resized is masked instead and earns no size credit. On a CIFAR ResNet one residual stream is owned
by 9–10 rows per pass; without the once-per-pass rule the walk re-cuts the same stream on every
owning row (the 13 Sep audit traced the −25 pp cliff on skinny ResNet-56 to exactly this:
8→7→6→5→4→3→2 channels on six consecutive rows), so the rule is part of the method, applied
identically to every compared policy.

### 3.2 State

The state is a variable-length sequence of per-layer tokens read by a small Transformer encoder
(3 layers, d = 256, 8 heads, no dropout) with a learned marker on the row **about to be pruned**, a
Graphormer-style attention bias on channel-coupling ids, sinusoidal positions and a pooled read-out
(mean over layers blended with the target token). Per-layer token contents (all database
z-scored except the bounded channels):

| Block | Channels | Content |
|---|---|---|
| Topology | 7 | layer family, kernel, width, stride, in/out features |
| Activation moments | 12 | statistics of the layer's activations on a fixed probe batch |
| Weight moments + filter-L1 shape | 12 + 7 | statistics of the weights and of the per-filter L1 distribution |
| Fortify | 4 | relative depth, stem flag, coupled flag, normalised width |
| Budget | 1 | fraction of parameters kept so far |
| **Slack** | 2 | **accuracy slack** \(\mathrm{clip}((\tau+\Delta acc)/\tau,-1,1)\) and pass progress |
| **Group cost** | 4 | parameter share and MAC share of the layer's whole group (all producers plus the slices every consumer and norm reads), owner count over the largest owner count in the net, structural cuts already applied to that group in this episode |
| Action cost | 2 × A | for the target row only: parameter and MAC fraction of the whole network each menu action would remove |

Slack and group cost are the two additions the v2 trains showed to be necessary. NEON's trichotomy
is scored on the *cumulative* Δacc against the origin, so the optimal policy is "cut while the band
has room, stop when it is spent"; without an explicit slack channel that rule is not a function of
the state. Group cost makes the cost side of every row observable: two layers with identical local
statistics can differ by an order of magnitude in what pruning them removes from the network,
and the schedule that reaches a deep operating point on a skinny ResNet (heavy cuts on wide,
cheap-in-accuracy block-internal convolutions; protection of the thin early streams) is expressible
only when the policy can see that cost.

The frozen `bert-base-uncased` encoder of the earlier design note (*Extending BERT Input
Mechanisms…*) is retained as an ablation flag; it tied the small Transformer on the hard held-out
net (ledger §16) and is not the default.

### 3.3 Action space

The menu is a small set of **(keep-rate, ranking)** pairs: identity, and the rates {0.9, 0.8} each
paired with two filter-importance criteria — L1 magnitude [18] and one partner criterion — five
actions in total. The paper agent's partner is FPGM (geometric-median distance, He et al. 2019);
BN-scale (Network Slimming, |γ|) and SVD (per-filter nuclear norm) are trained as sibling agents
and compared in §5. The rate is applied to the *alive* width with whole-channel rounding
(`round(rate·alive)`, never a no-op: any rate < 1 removes at least one channel); the ranking decides
which channels of the group survive, with every producer voting on the shared channel index after
per-layer max-normalisation. Stem rows and layers at ≤ 2 channels are identity-only.

Finer rungs are deliberately absent: on widths ≤ 16 a 0.95 rounds onto 0.9 or 0.8, so a finer
ladder adds actions without adding reachable widths (audit §2.1). A larger flat menu over four
rankings was also rejected for the paper agent: L1, L2 and SVD are near-collinear criteria, and
correlated arms dilute credit assignment without adding information (see §7 for the factored
alternative).

### 3.4 Reward

NEON's preference-aware trichotomy [1] on the **realised** cut: with Δ the cumulative validation
accuracy change (pp) against the unpruned network and \(r\) the percentage of parameters removed by
this step,

\[
R = \begin{cases} +r^{3} & \Delta > 0 \\ +r & -\tau \le \Delta \le 0 \\ -r^{3} & \Delta < -\tau \end{cases}
\qquad \text{learning signal } \tilde R = \operatorname{sign}(R)\,|R|^{1/3}.
\]

The cube-root is a strictly monotone rescaling — the ordering of steps is NEON's — that puts the
signal in percentage-point units so the critic can regress it. The nominal-rate form (score
\(100(1-\text{rate})\) regardless of what the group edit removed) and the raw cubic scale were both
run as controls: nominal-raw collapsed the policy (audit Part II/III; the in-band arm is invisible
next to the cubic arms once returns are normalised) and is reported as an ablation, not a paper
agent. Masked (non-structural) edits inside the band earn no size credit. Keeping a skip connection
is a hard constraint of the group rebuild, not a reward term.

### 3.5 Architecture

`NetworkEnv` (prune → fine-tune → evaluate), the `torch.fx` channel-group library, the state
builder, and a PPO actor–critic (separate encoders, MLP heads 300-300, zero-initialised policy
output so training starts from an exactly uniform policy). Legal-action masking is applied to the
policy distribution at sampling and at update time. The agent is ≈ 2.7 M parameters.

### 3.6 Training

**Catalog.** 24 pretrained CIFAR-scale networks (`configs/database_offline_wide.json`): thin
ResNet-20/56 at widths 6–16, chenyaofo ResNet-20/32/56, VGG-11/13/16-BN, MobileNet-v2 ×0.5/×1/×1.4,
DenseNet-BC-40, on CIFAR-10, SVHN and Fashion-MNIST. Every held-out cell of §5 (similar, unlike,
skinny-thin, CIFAR-100, ImageNet) is disjoint from this catalog by construction
(`configs/offline_pools_manifest.json`).

**Optimiser.** PPO over batches of four whole episodes: GAE (λ = 0.95, γ = 0.99) with a
bootstrapped critic, rewards scaled by the running standard deviation of discounted returns,
batch-normalised advantages, four clipped epochs (ε = 0.2) per batch with an approximate-KL stop,
gradient clipping 0.5, Adam 3·10⁻⁴ for the agent (decoupled from the fine-tune optimiser), entropy
bonus 0.01 annealed to 0.005 over 300 episodes. Each episode walks one catalog network for two
passes with full-network fine-tuning after every structural cut (Adam, 12 epochs, patience 4
during training; the evaluation protocol uses 40 / 10 for every method, so the policy trains under
a harsher recovery than it is tested with).

**Selection and stopping (the learning governor).** The training-side score of an episode is
\(1-\text{kept}\) at the deepest point whose validation Δacc is still inside τ — the same object the
evaluation protocol quotes. Snapshots are selected on a **fixed deterministic probe**: every 12
on-policy episodes the current policy is run argmax on two fixed catalog networks
(`resnet56-width6`, `resnet20-width10`) and the mean probe score, not the noisy on-policy batch
mean, decides `latest_best`, freezes a snapshot (actor, critic, standardizer, action/state contract)
and drives patience. Training never stops before 250 episodes, then stops after 150 episodes
without probe improvement or at a 6-day fuse. When the probe has not improved for 50 episodes the
trainer **rewinds** to the elite weights (actor and critic), resets Adam and raises the entropy
bonus to 0.02 for 30 episodes (at most three times) — a return-then-explore restart from the elite
in the sense of population-based training's exploit step and Go-Explore, used as search, never as
evidence that the policy improved. This replaces the earlier rule "stop after 100 episodes without
beating the best 4-episode batch mean", which the audit showed to terminate a still-learning policy
on a lucky order statistic.

**Contract.** Every checkpoint directory carries `policy_config.json` (menu, per-action rankings,
state channels, marker alignment, passes) and `standardizer.pt`; the evaluation runner pins that
contract before it builds its configuration, so a frozen actor is always replayed under the state
and menu it was trained with.

### 3.7 Complexity

Wall-clock is dominated by fine-tuning: a 24-net training episode (two passes, 12-epoch FT) takes
≈ 11–25 min on an RTX-4090-class GPU; a held-out evaluation walk with the 40-epoch protocol takes
1–6 h per network (DenseNet-100 ≈ 2 days). Deterministic probes add two episodes per twelve. Image
datasets and deep CNNs remain substantially heavier than NEON's tabular DNNs (proposal limitation;
still true). Training runs single-GPU; two-plus GPUs do not halve the sequential prune+FT chain.

---

## 4. Evaluation

NEON §4: algorithms, setup, results, discussion. SPECTRA splits setup here and numbers in §5.

### 4.1 Compared methods

Every method below shares the environment: the same channel-group rebuild, the once-per-pass rule,
the same L1 (or per-action) ranking, the same fine-tune (Adam, 40 epochs, patience 10), the same τ
and the same trajectory protocol. They differ only in the rate picker.

| Method | What it chooses | Same loop? |
|---|---|---|
| **SPECTRA (DRL, v3)** | (rate, ranking) per row from the learned policy, argmax at test | yes |
| Greedy (“l1-once”) | strongest legal cut (0.8) on every unlocked row | yes |
| Mild (“mild-once”) | 0.9 on every unlocked row | yes |
| Random | uniform over legal cuts | yes |
| Ranking ablations | greedy / mild with `SPECTRA_FILTER_IMPORTANCE` ∈ {l2, svd, fpgm, bn_scale} | yes |
| Frozen 10-net actor (Path 3) | provenance row: argmax ≡ mild (audit F4) | yes |

Group-once is applied to all of them: the plain-walk heuristics (H0 mild-plain, l1-plain; ledger
§85, §88) are reported once to show that the once-per-pass rule, not the rate picker, sets the
easy-net plateau. DepGraph [12] / SPA [81] / OCS [93] / SACP [94] are quoted on the Pareto with a
different-fine-tune caption; they are not same-loop baselines and SPECTRA does not claim to beat
them on their home cells (Gilad 18 Aug).

**Figure (NEON Fig. 5 analog):** per architecture · dataset, TEST Δacc vs parameters kept and vs
MACs kept at the quoted operating point of every method; the coverage matrix remains a separate
genericity map.

### 4.2 Protocol

- **Trajectory protocol.** No identity-pad and no size stop. A held-out network is walked for
  the same number of passes the actor was trained with (two); after every structural cut the
  validation and test loaders are scored. The operating point quoted is **`val_best`**: the most
  compressed point whose *validation* Δacc is within τ = 10 pp; its **test** Δacc is reported but
  never used for selection. `floor_hold` / `floor_cross` (the last point ≥ 0.70 and the first
  below) are labels for size-matched comparison, not stops. Terminal points with validation over τ
  are not operating points.
- **Counts.** Parameter and MAC fractions are exact ratios of counts (the earlier three-decimal
  megaparameter counter quantised skinny nets to 0.2 steps and is not quoted).
- **Deterministic evaluation.** Argmax over the legal-masked policy; encoder in inference mode.
- **Win criterion (per network, against the same-loop heuristic at its own `val_best`):** kept
  fraction ≤ the heuristic's at equal-or-kinder test Δacc, or ≥ 2 pp kinder test Δacc at equal
  kept fraction. Unmatched keep is reported as a different Pareto point, not as a win.
- **Datasets in DRL train:** CIFAR-10 [73], SVHN [74], Fashion-MNIST [72]. CIFAR-100 [73] and
  ImageNet [71] are held-out transfer cells (no DRL training on either).
- **Held-out cells:** similar families (thin r20-w16, r56-w10, r44, VGG-19-BN, MobileNet-v2×0.75,
  DenseNet-100), unlike families (ShuffleNet-v2 ×1/×1.5, RepVGG-A0/A1), skinny-thin
  (r20-w2, r56-w4), CIFAR-100 (5 nets), ImageNet (frozen probe). Skip akamaster ResNet-32.
- **Quote `eval_test` TRAJ lines only.** Log `eval_train` scores the CNN's training loader and is
  never a result.

---

## 5. Evaluation results — fill from the ledger

Proposal §4 (NEON on flattened Fashion-MNIST / CIFAR / SVHN) is **motivation**, not a SPECTRA table. Do not mix those DNN-on-pixels numbers into §5.

### 5.1 Similar-family transfer (C10) — LOCKED

See ledger §3. Easy nets (VGG-19 BN [76], DenseNet-100 [4], ResNet-44 [3], MobileNet-v2×0.75 [78], thin ResNet-20 w16) stay inside τ=10 on seeds 42/43/44. Thin ResNet-56 w10 is the similar-family miss and is seed-sensitive (−9.2 / −12.4 / −13.0). Chain A argmax **20945570** (ledger §57, PRELIM `det=1`): r20-w16 **−5.1 @ 0.669/0.649** inside unmatched vs sampled s42 **−5.4 @ 0.603/0.639**; r56-w10 **−12.8 @ 0.661/0.421** **miss** vs sampled s42 **−9.2 @ 0.658/0.488** (ties mild at the same size); r44 **−4.5 @ 0.699/0.542** inside unmatched vs sampled s42 **−4.3 @ 0.632/0.519** (ties mild **−4.7 / −4.1 / −4.2 @ 0.699/0.542**); VGG-19 **−2.7 @ 0.699/0.738** inside unmatched vs sampled s42 **−2.7 @ 0.879/0.882** (same Δacc, fewer params); MobileNet **−2.4 @ 0.688/0.587** inside near-matched vs sampled s42 **−2.5 @ 0.689/0.630**; DenseNet **−2.1 @ 0.801/0.803** inside unmatched vs sampled s42 **−2.0 @ 0.847/0.849**. Do not overwrite §3. Catalog **COMPLETED**. C100 argmax **20945572 COMPLETED**: r20-w16 **−16.2** miss; RepVGG **−11.3 @ 0.685** miss (§58). Eval-only FLOP floor 0.70 on the same frozen actors (ledger §29, PRELIM): r56-w10 s42 **−5.9 @ 0.902/0.702** and s44 **−5.6 @ 0.911/0.702** inside τ at ~90% params; s43 **−14.3 @ 0.946/0.700** still misses at 95% params. DenseNet **−2.2 / −2.4 / −2.2** @ 0.837/0.833, 0.822/0.827, 0.780/0.797. Operating point, not a three-seed rescue of the similar r56-w10 miss. FLOP+prefer (ledger §33): r56-w10 **−3.8 / −4.0 / −4.4 @ 0.702/0.872** **LOCKED three-seed inside τ**; r20 **−3.8 / −3.5 / −3.8 @ 0.717/0.880**; r44 **−2.3 / −2.6 / −2.6 @ 0.702/0.872**; VGG-19 **−2.2 / −2.6 / −2.4 @ 0.837/0.923**. DenseNet **−2.0 / −1.9 / −2.7 @ 0.870/0.951** **LOCKED three-seed**. MobileNet **−1.9 / −2.0 / −2.2 @ 0.767/0.912** **LOCKED three-seed**. Prefer is the lever; FLOP-floor-only s43 was **−14.3 at 95% params**. Look-ahead greedy (ledger §34, PRELIM): r20 three-seed **−7.4 / −7.0 / −7.3 @ 0.713/0.525** inside; r56-w10 three-seed **−22.4 / −20.1 / −22.2 @ 0.702/0.376** cliff; r44 **−8.0 / −9.0 / −8.0 @ 0.703/0.391** inside τ; VGG-19 **−3.5 / −3.2 / −2.5 @ 0.703/0.669**; MobileNet three-seed **−3.2 / −3.3 / −3.0 @ 0.708/0.511** (s42 0.938→0.906) inside at 51% FLOPs vs prefer 91%; DenseNet three-seed **−2.4 / −2.6 / −2.6 @ 0.701/0.679** (s42 0.949→0.925) inside. Catalogs COMPLETED. Same-loop FLOP+prefer greedy s42 (**20715876 COMPLETED**, ledger §48, PRELIM): DenseNet **−2.1 @ 0.870/0.951** near-tie vs DRL prefer **−2.0** same size; r56 **−4.4 @ 0.702/0.872** vs DRL **−3.8** (prefer stopped the look-ahead cliff). Same-loop FLOP-floor greedy s42 (**20715868 RUNNING**, ledger §51, PRELIM): r56 **−9.2 @ 0.952/0.702** inside, size-matched to FLOP-floor look-ahead three-seed **−8.3 / −9.2 / −8.6**; unconstrained greedy was **−23.1**. Do not lock.

**Figure TBD:** bar or table of Δacc vs params kept, three seeds.

### 5.2 Unlike-family transfer (C10) — LOCKED (n=3 seeds)

See ledger §4. ShuffleNet-v2 and RepVGG, never in train, all inside τ=10 on seeds 42/43/44. RepVGG-A0 is −4.8 / −4.6 / −4.8 pp at 0.681/0.565 (same size). RepVGG-A1 is −4.7 / −4.4 / −4.3 at 0.650/0.521 (same size). ShuffleNet logs a mask fallback on some layers; quote TEST Δacc, do not claim every ShuffleNet layer was structurally resized. Eval-only FLOP floor 0.70 on unlike (**LOCKED three-seed**, ledger §28): ShuffleNet×1 **−1.9 / −1.8 / −1.5**; ×1.5 **−2.0 / −2.3 / −2.8**; RepVGG-A0 **−3.9 / −4.6 / −4.6**; A1 **−4.3 / −4.2 / −4.2**. Milder operating point; default unlike was already inside τ. FLOP+prefer on the same frozen actors (**LOCKED three-seed**, ledger §30): ShuffleNet×1 **−1.5 / −1.1 / −1.5 @ 0.871/0.944**; ×1.5 **−2.1 / −2.1 / −1.9 @ 0.879/0.950**; RepVGG-A0 **−4.4 / −3.7 / −4.0 @ 0.715/0.756**; A1 **−3.5 / −3.5 / −4.1 @ 0.705/0.753**. Same size. FLOP-floor **look-ahead** three-seed (ledger §47 / §65, PRELIM): ShuffleNet×1 **−1.7 / −2.0 / −2.2 @ 0.801/0.716**; ×1.5 **−2.5 / −2.6 / −2.4 @ 0.771/0.701**; RepVGG-A0 **−6.9 / −6.6 / −7.1 @ 0.847/0.701**; A1 **−6.7 / −5.7 / −6.2 @ 0.857/0.702** — **same size**, heuristic. FLOP-floor **greedy** s42 (ledger §64, PRELIM): ShuffleNet×1 **−2.0 @ 0.801/0.716**; ×1.5 **−2.4 @ 0.771/0.701**; RepVGG-A0 **−6.7 @ 0.847/0.701**; A1 **−6.5 @ 0.857/0.702** — **same sizes as §47**; heuristic.

### 5.3 Held-out thin ResNets and learned vs greedy — LOCKED

See ledger §5–6 and catalog ladder §17. Easy r20-w2: DRL ≈ greedy [18] at 60% params (s43 **−4.9 @ 0.600/0.760**; look-ahead greedy r20 **−4.0 @ 0.600/0.753**; FLOP-floor s42 **−2.1 @ 0.600/0.773**, s43 **−3.7 @ 0.600/0.763**; s44 FLOP-floor r20 **−4.0 @ 0.800/0.799**, a larger net). Chain A argmax **20945568** r20 **−4.4 @ 0.600/0.741** vs sampled **20945567 −3.9 @ 0.600/0.748** vs locked s42 **−4.2 @ 0.600/0.760**. Hard r56-w4 argmax **−25.2 @ 0.667/0.465** vs sampled **−25.4 @ 0.667/0.482** vs locked s42 **−15.9 @ 0.704/0.550** (ledger §54.5). The **policy** is the cliff at 0.667 params; locked −15.9 was a milder *sample*. Do not overwrite that LOCKED row. Hard r56-w4: C10-thin-only and encoder A/Bs stay near −24 pp; 10-net DRL **−15.9 / −16.2 / −17.2** at **identical** 0.704/0.550 (three seeds, **misses τ**). **24-net s42 (20201263) is worse: −25.0 @ 0.704/0.499.** Unmatched greedy ~−24 at 0.667. Param-floor look-ahead greedy **−24.9 @ 0.722/0.477** (job 20213131). Eval-only FLOP floor 0.70 on the **same frozen 10-net actors**: s42 **−8.9 @ 0.907/0.702**, s43 **−9.2 @ 0.926/0.703**, s44 **−9.6 @ 0.907/0.703** — all **inside τ=10**, at ~91–93% params / 70% FLOPs (**LOCKED** three seeds). Prefer-Δparams/ΔFLOPs under that FLOP floor (**LOCKED three seeds**): r56-w4 **−8.9 / −8.0 / −8.1 @ 0.704/0.872**; r20-w2 **−2.7 / −2.7 / −1.9 @ 0.600/0.886**. Similar r56-w10: DRL −13.0 vs greedy −23.1 at ~0.60 params. AMP, skinny-in-train, and budget-in-state did not move r56-w4 (ledger §18). Param floor 0.80 (no FLOP floor) still cliffs: **−21.7 / −25.7 / −22.5** at ~0.80 params / 0.54 FLOPs (ledger §24). Same-loop L2 ranking three-seed: r56-w4 **−24.1 / −21.2 / −23.4** (unmatched sizes). SVD s42 **−25.6 @ 0.667/0.482**; SVD s43 **−19.8 @ 0.593/0.475** (same size as L2 s43 −21.2). SVD s44 **−23.7 @ 0.685/0.490** (same size as L2 s44 −23.4). Keep L1 as the default ranker. FPGM ranking A/B (ledger §59, PRELIM catalog **COMPLETED**): r20-w2 **−4.8 @ 0.600/0.741** same size as L1 argmax **−4.4**; r56-w4 **−23.4 @ 0.667/0.465** vs L1 **−25.2** same size (1.8 pp milder, still a cliff). BN-scale (ledger §60, PRELIM): r20-w2 **−3.8 @ 0.600/0.741** same size as L1; r56-w4 **−27.1 @ 0.667/0.465** — **worse** than L1. Do **not** retrain the frozen actor.

### 5.4 CIFAR-100 recoverability — LOCKED VGG / LOCKED tiny-cut others

See ledger §7. **CLAIM:** C100 is not a second genericity table until a 5–10% structured cut recovers on more than VGG [76]. Probe 20204214 is complete. VGG-11 BN recipe is two-run (TEST gains at 90–95% params). Residual keep-rate 0.8 still leaves ~95–96% params. DenseNet-40 keep 0.8 left 99.0% params (TEST −0.43). MobileNet-v2×1 keep 0.8–0.9: TEST gain at ~98–99% params but **val DROP** (~−12.5 pp). Do not describe early C100 failure as “the C10 agent was not trained on C100.” That failure mode was already visible in the proposal’s NEON-on-images C100 table (very low absolute accuracy); SPECTRA’s CNN probes show the *structured* version of the same hardness.

### 5.5 24-net catalog — LOCKED miss (r56-w4) / PRELIM similar + unlike

Jobs 20201235 / **20202693 COMPLETED** / 20204215. Question for skinny r56-w4: does 24-net move past −15.9/−16.2/−17.2? **No. 20201263 COMPLETED** 22:30: **−25.0 @ 0.704/0.499** (same params as 10-net s42, fewer FLOPs). Easy r20-w2 **−5.7 @ 0.600/0.748**. Catalog diversity 3-net→10-net was the gain; 10-net→24-net is a miss. Dedicated similar skip-train **20201260 COMPLETED** 21:08 (24-net s42): r20-w16 **−5.9 @ 0.695/0.611**; r56-w10 **−12.6 @ 0.634/0.410** miss vs 10-net s42 **−9.2 @ 0.658/0.488**; r44 **−4.5 @ 0.682/0.542**; VGG-19 **−2.8 @ 0.817/0.771**; MobileNet-v2×0.75 **−1.7 @ 0.668/0.651**; DenseNet-100 **−1.7 @ 0.835/0.831**. Skip r32. 24-net s44 (**20204215 COMPLETED** 01:09): r56-w10 **−8.6 @ 0.616/0.538** inside τ vs 10-net s44 **−13.0 @ 0.604/0.397** (not size-matched); r44 **−3.3 @ 0.693/0.576**; VGG-19 **−2.6 @ 0.752/0.801**; MobileNet **−2.2 @ 0.721/0.630**; DenseNet-100 **−2.4 @ 0.798/0.813** vs 10-net s44 **−2.1 @ 0.834/0.851**. s42/s43 r56-w10 still miss. Unlike **20201265 COMPLETED** (ledger §27): RepVGG-A0 **−4.9 @ 0.671/0.545** vs 10-net s42 **−4.8 @ 0.681/0.565**; A1 **−3.9 @ 0.662/0.552** vs 10-net **−4.7 @ 0.650/0.521**; ShuffleNet **no TEST** (grouping / finetune fail). Do not quote the job-mean −0.02 pp.

### 5.6 Frozen 10-net → CIFAR-100 (claim C9) — LOCKED mixed

See ledger §21. Same actors as §5.1–5.3, no extra agent training. VGG-16 BN [76] stays inside τ=10 (−7.5 / −7.8 / −7.3 pp at ~80–83% params). Thin ResNet-20 w16 and ResNet-56 w15 miss under Adam-40 (≈ −14 to −19 pp). RepVGG-A0 misses (≈ −11 to −13 pp). ShuffleNet-v2×1 s42 **−3.9 @ 0.833/0.823** (job 20289197); s43 **−3.4 @ 0.819/0.823** (job 20307286); s44 **−4.3 @ 0.837/0.823** (job 20307395; masked effective-params 0.815 — quote 0.837). Three-seed inside τ. Read with §5.4: under the C9 recipe, the families that recover from a structured cut are the families this frozen agent can transfer to. Chain A C100 argmax **20945572** (ledger §58, PRELIM `det=1`, catalog **COMPLETED**): r20-w16 **−16.2 @ 0.669/0.649** **miss** vs sampled s42 **−19.3 @ 0.604/0.645** (ties mild **−17.0 / −16.0 / −16.1 @ 0.669/0.649**); r56-w15 **−17.9 @ 0.689/0.499** **miss** vs sampled s42 **−15.0 @ 0.694/0.600** (ties mild **−18.2 / −16.8 / −17.7 @ 0.689/0.499**; still beats look-ahead **−34.6**); VGG-16 BN **−8.3 @ 0.769/0.771** **inside** vs sampled s42 **−7.5 @ 0.797/0.834**; ShuffleNet-v2×1 **−6.6 @ 0.726/0.718** **inside** vs sampled s42 **−3.9 @ 0.833/0.823** (quote structural keep); RepVGG-A0 **−11.3 @ 0.685/0.556** **miss** vs sampled s42 **−12.1 @ 0.571/0.446** (ties mild **−11.8 / −11.5 / −11.8 @ 0.684/0.548**). Mixed split is policy-grade. Do not overwrite §21.

SGD-recipe A/B (ledger §25, PRELIM three-seed): same frozen actors, 80-ep SGD+cosine+MixUp+AutoAugment. r56-w15 **−9.1 @ 0.612/0.442** / **−9.4 @ 0.672/0.450** / **−10.8 @ 0.571/0.360** — s44 misses τ at a smaller net. r20-w16: s42 **−12.2** miss, s43 **−10.1** miss, s44 **−8.8 @ 0.698/0.613** inside τ. FLOP floor 0.70 on that recipe (three-seed): r20 **−5.7 / −7.0 / −7.7** inside; r56 **−4.7 / −10.4 / −6.4** — s43 **misses at 95% params / 70% FLOPs**. FLOP+prefer (same recipe): r56 **−8.2 / −4.5 / −8.5 @ 0.703/0.872** three-seed inside τ; r20 s44 **−10.4 @ 0.716/0.879** miss. C9 Adam-40 FLOP+prefer (ledger §35, **LOCKED mixed**): three-seed VGG **−7.6 / −7.7 / −7.3 @ 0.838/0.931** inside; ShuffleNet **−3.5 / −4.1 / −4.6 @ 0.873/0.944** inside; residuals **−14.5 / −13.8 / −12.7** and **−12.0 / −11.4 / −10.5** miss; RepVGG **−13.0 / −12.5 / −13.0 @ 0.719/0.756** miss at matched sizes. Random (ledger §40, PRELIM) tracks the same split: VGG three-seed unmatched **−9.3 / −8.5 / −8.5** inside; ShuffleNet two-seed inside unmatched; residuals/RepVGG miss (s44 RepVGG **−12.6 @ 0.664/0.498**). FLOP-floor only (ledger §36, PRELIM s42/s43/s44 catalogs COMPLETED): VGG **−8.4 / −8.4 / −7.7** inside unmatched; ShuffleNet **−3.4 @ 0.881/0.858 / −3.9 / −4.0** inside unmatched (quote structural keep); RepVGG s42 **−10.2 @ 0.886/0.761** miss vs s43/s44 **−8.2 / −7.4** inside only at ~95% params; r20 **−11.8 / −12.1 / −10.6** miss unmatched; r56 **−10.4 / −11.7 / −12.0** miss unmatched. Prefer still misses RepVGG at 72%/76%. Look-ahead greedy (ledger §37, PRELIM): s42/s43/s44 catalogs COMPLETED; r20 three-seed **−18.1 / −17.3 / −17.9 @ 0.716/0.525** miss; r56 three-seed **−34.6 / −32.1 / −33.2 @ 0.701/0.357** cliff; VGG three-seed **−8.8 / −8.6 / −9.0 @ 0.701/0.672** inside; ShuffleNet three-seed **−6.2 / −6.4 / −5.9 @ 0.736/0.682** inside (quote structural keep); RepVGG three-seed **−11.4 / −11.7 / −12.1 @ 0.709/0.577** miss vs prefer **−12.5 / −13.0 @ 0.719/0.756**. Mild (ledger §38, PRELIM catalogs COMPLETED): r20 three-seed **−17.0 / −16.0 / −16.1 @ 0.669/0.649** miss; r56 three-seed **−18.2 / −16.8 / −17.7 @ 0.689/0.499** miss; VGG three-seed **−8.0 / −8.3 / −8.0 @ 0.811/0.822** inside; ShuffleNet three-seed **−4.0 / −4.4 / −4.3 @ 0.860/0.835** inside (quote structural keep); RepVGG three-seed **−11.8 / −11.5 / −11.8 @ 0.684/0.548** miss (tracks default family split). Do not overwrite §21.

C100-trained DRL actor on the same held-out residuals (ledger §31, PRELIM one seed, job **20353582**): r20-w16 **−8.3 @ 0.673/0.627**; r56-w15 **−8.4 @ 0.662/0.469**. Both inside τ. Not size-matched to §21. Do not quote the DRL train returns. s43 residual eval **20884674 COMPLETED** (ledger §56, PRELIM **sampled**, no `det=1`): r20-w16 **−16.4 @ 0.597/0.611** miss unmatched vs §31; r56-w15 **−7.5 @ 0.623/0.495** inside τ unmatched vs §31 **−8.4 @ 0.662/0.469**. Do not treat as a second seed. Do not overwrite §31. Matched-VGG residual eval s42 **20884672 COMPLETED** (ledger §61, PRELIM **sampled**): r20-w16 **−12.0 @ 0.612/0.598 miss**; r56-w15 **−10.4 @ 0.697/0.402 miss** unmatched vs §31. Both residuals miss. s44 recoverable train **20884675 COMPLETED** — do not quote train.

### 5.7 Digit-MNIST LeNet (held-out dataset) — LOCKED three seeds

See ledger §22. Never in the 10-net train catalog (Fashion-MNIST was). TEST **+2.8 / +2.7 / +2.9 pp**. Toy 1-channel net, modest param cut. NEON-style cheap dataset cell, not ImageNet.

### 5.8 SVHN ResNet-20 width 8 (held-out width) — LOCKED three seeds

See ledger §23. SVHN [74] is in the 10-net train mix; this is a **new width**, not a held-out dataset. TEST **−2.0 / −1.5 / −2.0 pp**, all inside τ=10. Do not quote in-catalog r20-w16 or VGG-11 SVHN as transfer.

### 5.9 Frozen 10-net → ImageNet MobileNet-v2 — PRELIM one seed

See ledger §41. Same frozen C10 actor, **no** ImageNet DRL train. Truncated-JPEG loader. s43 **20360208** TEST **−4.6 @ 0.823/0.729** (71.9% → 67.3%), inside τ=10. s42 TIMEOUT (no TEST). s44 running. Origin 71.9% is this loader’s unpruned `eval_test`, not a literature ImageNet number. Do not quote 82.8% or train-loader. Probe sentence, not a home-court SOTA claim vs DepGraph / OCS.

---

## 6. Discussion — DRAFT

**NEON’s pruning strategy, CNN version.** The agent learns *where* to cut. On easy C10 nets the strongest legal cut [18] is already good, so DRL ties greedy. On skinny-deep ResNets the schedule matters, but the environment is still harsh: residual groups decide params/FLOPs, and a 0.8 rate is not a 20% size cut [3]. Catalog diversity **3-net → 10-net** moved the hard thin ResNet-56 (−24 → −15.9). **24-net did not** (ledger §17: **−25.0 @ 0.704/0.499**). Encoder width did not.

**Mechanism vs policy.** DepGraph [12] and SPA [81] are the right papers to thank. SPECTRA should not claim “first grouping for any CNN.” It should claim “first (in this lineage) offline preference-aware *policy* for structured CNNs,” with honest held-out splits.

**Advisor 18 Aug — comparison and Pareto.** Results are compared to SOTA / similar papers as a reference, not as a home-court contest. The justification for a generic DRL agent is transfer **without** per-target agent training or adaptation. Claiming to beat a focused method on its own architecture × dataset would be mind-boggling; hope is not a claim. Maintain a NEON-style Pareto (compression vs TEST Δacc vs heuristics) in addition to the coverage matrix. ImageNet is frozen transfer from CIFAR-10 / CIFAR-100, not DRL training. Details: [GILAD_DIRECTIVES_18AUG.md](GILAD_DIRECTIVES_18AUG.md).

**Why C100 is a different chapter — and what “solve” means.** Recoverability probes have no agent. If fine-tune cannot undo a mid-layer prune, A2C cannot learn a useful C100 policy. VGG-11 BN under SGD+aug 160 ep *can*; thin C100 ResNets at keep-rate 0.8 still keep ~95% of parameters (ledger §7.2). Mixing those nets into a C10 agent is how train returns go to −100, not how dataset transfer is demonstrated. Frozen 10-net → C100 TEST (ledger §21 / argmax §58) matches that split under Adam-40: VGG-16 BN and ShuffleNet stay inside τ=10; thin ResNets and RepVGG-A0 miss. Changing the *fine-tune* (ledger §25) can move r56-w15 inside τ while r20-w16 still misses. The 3 Sep root cause is sharper than “C100 is hard”: under NEON’s `−reduction³`, a net whose every real cut busts τ has an empty reward band, so the training argmax is “never prune” (42/42 C100 steps over-budget, job **20930175**). That is **not** class count (spoof §55; ImageNet 1000-way already transfers). Prefer **solving** that band (graded overshoot reward, a fine-tune that reopens the band, admit-only-recoverable nets) over stopping at a limitation caption — but the mixed C9 split is already writeable if the solve does not land before freeze. The proposal already flagged C100 as the hardest image set for a generic pruner; the CNN experiments refined that to *family-wise recoverability plus a non-degenerate reward*, not “needs BERT.”

**Catalog size is not the remaining lever in this thesis.** Diversity 3 → 10 moved the hard thin ResNet-56. 24 CIFAR cousins reversed that gain. Empty-band nets teach identity. So the paper’s 10-net agent is the right *scientific* object: it leaves ShuffleNet, RepVGG, CIFAR-100, and ImageNet as transfer evidence. A large shelf-product catalog is a different experiment (new hold-out, band screen, weeks of GPU) and does not replace those tables.

**Limitations (write them):** ImageNet is a **frozen-transfer probe** (two-seed unmatched PRELIM, truncated JPEG; ledger §41), not DRL train and not a SOTA ImageNet fight; ShuffleNet grouping incomplete (mask fallback); r56-w4 misses τ=10 at the 0.70-param operating point (sampled 10-net DRL −15.9; **argmax −25.2 @ 0.667** — the policy is the cliff; **24-net DRL −25.0 @ 0.704/0.499**); FLOP-floor 0.70 enters τ at ~91–93% params / 70% FLOPs (s42 −8.9, s43 −9.2, s44 −9.6); prefer-Δparams/ΔFLOPs is a **heuristic** that puts the same net inside τ at **0.704 params / 0.872 FLOPs**; C9 Adam-40 misses on C100 thin ResNets (SGD-80 one-seed puts r56-w15 inside τ, r20 still miss); Places365 [75] and GoogLeNet [77] unused. Cheap-dataset transfer that *is* in: digit-MNIST LeNet TEST gain; SVHN r20-w8 inside τ (width held out, dataset in train). Do not quote overnight-matrix −1.2 pp or eval_train as TEST (ledger §10).

---

## 7. Conclusions and future work — DRAFT

SPECTRA shows that NEON's offline, preference-aware DRL protocol [1] extends to structured CNNs:
one agent, trained on a catalog of CIFAR-scale networks, is applied frozen to similar and unlike
families, to skinny widths it never saw, and to held-out datasets, under one loop that is identical
for the agent and for every heuristic it is compared with. The grouping problem that used to look
like part of the thesis is professionally handled by DepGraph [12] and SPA [81]; that is a gift.
The remaining question is the one the proposal posed — whether a single preference-aware agent can
carry a pruning *schedule* across CNN families and datasets — and §5 answers it cell by cell with
the v3 agent's `val_best` operating points against the same-loop heuristics (fill from the ledger;
the frozen 10-net rows stay as provenance).

**Future work — V4 candidates, in the order we would run them if the v3 TESTs do not clear the
win criterion (for Gilad to approve; not this freeze's GPU plan).**

1. **Factored action head (rate × ranking) — implemented and queued (16 Sep, `offline_train_v4_factored`).**
   Two Categorical heads — rate ∈ {1.0, 0.9, 0.8} and ranking ∈ {L1, FPGM, BN-scale, SVD,
   Taylor} — with the joint log-probability as their sum and the ranking head inactive on
   identity. Each head receives the full sample count, so correlated criteria stop diluting
   credit; Taylor (first-order |w·∂L/∂w| on a training batch, bound before each cut) is the one
   data-dependent criterion. It is the principled version of "one agent for all rankings" and
   the natural successor of the three sibling agents of §3.3. A second copy adds a stricter
   training band (τ_train = 6) as a curriculum. Flag-gated (`SPECTRA_FACTORED_HEAD`); the v3
   agents are unchanged.
2. **Merging the sibling agents at evaluation.** Two honest variants: (a) a *portfolio* — run
   each sibling's trajectory and select the actor per network on **validation** `val_best`
   (a light per-network adaptation; label it as such on the Pareto); (b) *distillation* — a
   student with the factored head trained to imitate the siblings' argmax choices on the catalog,
   then fine-tuned with PPO. Averaging the siblings' action probabilities is not one of them:
   their action indices mean different criteria.
3. **Train-only τ curriculum** (`SPECTRA_TRAIN_TAU`, implemented, off): a stricter band during
   training so the band edge is reached on every catalog network, with slack and reward both
   τ-relative so the policy transfers to the evaluation τ unchanged. To be used if two passes
   still leave the edge rare (v2: 2.5 % of steps).
4. **Per-layer sensitivity proxy in the state:** the validation Δacc caused by the last cut of
   each group, so allocation can condition on observed fragility rather than on cost alone; and
   refreshed activation moments for every layer a group edit touched.
5. **BN-recalibration step proxy.** Score each step with BN-statistics recalibration (a forward
   pass, no backward) and fine-tune only at labelled points; ~50× cheaper steps, thousands of
   episodes per GPU-day. Requires a calibration table (BN-recal Δacc vs fine-tune Δacc on the
   catalog) before it can replace the per-step signal.
6. **Self-imitation of elite walks** (SIL / AWR) as the alternative to the elite-weights rewind,
   with an importance-corrected buffer of high-`val_best` episodes; the honest research sentence
   for "keep learning after the first peak" if the rewind proves too greedy.
7. **Larger band-screened catalog** (100–150 nets across the 287-file pool, new hold-out cells of
   the same roles), only after a peaked policy exists; the 24-net catalog is the thesis object.
8. **Preference sweep as product knobs** (τ ∈ {5, 10, 15}; FLOP-matched labels) — NEON's Pareto
   family, not a new architecture search. **No ImageNet DRL train** (Gilad 18 Aug); broader
   *frozen* ImageNet transfer stays in scope as coverage.

Closed lines (do not retry as future work): encoder / BERT / AMP / skinny-in-train A/Bs
(ledger §16–§18); finer rate rungs (audit §2.1); Taylor ranking without a cost calibration;
warm-starts from any pre-v2 actor. Michael Bohadana's NAP2 is NAS performance prediction, not a
SPECTRA lift.

---

## Appendix A — Protocol archaeology (8–16 Aug) — DRAFT

Numbers in [RESULTS_LEDGER.md](RESULTS_LEDGER.md) §§10–20. Do not promote these to §5 tables except where already LOCKED (C7, floor, Fortify).

**A.1 Inherited bugs.** Until the overhaul, `torch.nn` prune left shapes unchanged, FT was `range(0)`, DDP nested every step, and episode return was the last reward. Later compression numbers are real rebuilds (ledger §11).

**A.2 Fine-tune mode.** Layer-only FT: 0/32 recoveries (proposal default freeze). Full-net 40 ep + rates 1.0/0.9/0.8 made C10 recoverable (ledger §12). Group-aware freeze did not.

**A.3 Eval floor.** Without 0.70, held-out r20-w2 is −13 to −15 pp at 40% params; unconstrained r56-w4 is −42 pp at 17% params (20140546). With the floor, 2-net structural TEST is −3.1 @ 0.60/0.79 (20066579). Overnight **−1.2 pp is not TEST**.

**A.4 Encoder ablation.** Ledger §16. All four encoders ~−24 pp on r56-w4. Catalog diversity is the lever (§17).

**A.5 Mixed C10+C100.** 45–48% train-within −10 was C100, not the DRL recipe (ledger §14). Split datasets.

**A.6 C100 Adam probes.** 20158277: 2/34 val-OK, both ≥98.5% params. 20168590 crop+flip: 1/26 at 0.995 params. Not a 2–5% menu. VGG SGD 160-ep is the first real cut (ledger §7.1).

**A.7 Code SHA / catalogs.** Night git `e985d5e`. Train: `database_offline_train.json` (10) then `database_offline_wide.json` (24, running). Pool: `offline_pools_manifest.json` (287 mapped, not all trained).

**A.8 Proposal vs implementation (one-page).** Kept: global-generic offline DRL, CNN meta-features, Transformer sequence over layers [35], NEON τ, structured rebuild, ResNet/DenseNet/VGG/MobileNet families [3, 4, 76, 78], C10/SVHN/Fashion train. Dropped or postponed: ImageNet/Places365 in the loop; 0.7/0.6 menu; pooling as an action; skip-ratio reward; nested NEON on FC; frozen BERT default; “first any-architecture grouping.”

---

## References

Proposal numbering **[1]–[78]** unchanged. Survey additions **[79]–[91]**.

[1] Hirsch, L., & Katz, G. (2022). Multi-objective pruning of dense neural networks using deep reinforcement learning. *Information Sciences*, 610, 381–400. https://doi.org/10.1016/j.ins.2022.07.134

[2] O’Shea, K., & Nash, R. (2015). An introduction to convolutional neural networks. arXiv:1511.08458.

[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.

[4] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *CVPR*, 4700–4708.

[5] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.

[6] Reed, R. (1993). Pruning algorithms — a survey. *IEEE Transactions on Neural Networks*, 4(5), 740–747.

[7] Anwar, S., Hwang, K., & Sung, W. (2017). Structured pruning of deep convolutional neural networks. *ACM JETC*, 13(3), 1–18.

[8] LeCun, Y., Denker, J., & Solla, S. (1989). Optimal brain damage. *NeurIPS*, 2.

[9] Hassibi, B., Stork, D. G., & Wolff, G. J. (1993). Optimal brain surgeon and general network pruning. *IEEE ICNN*, 293–299.

[10] Molchanov, P., Tyree, S., Karras, T., Aila, T., & Kautz, J. (2016). Pruning convolutional neural networks for resource efficient inference. arXiv:1611.06440.

[11] Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2018). Rethinking the value of network pruning. arXiv:1810.05270.

[12] Fang, G., Ma, X., Song, M., Mi, M. B., & Wang, X. (2023). DepGraph: Towards any structural pruning. *CVPR*, 16091–16101. arXiv:2301.12900.

[13] Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural network. *NeurIPS*, 28.

[14] Chen, X., Zhu, J., Jiang, J., & Tsui, C. Y. (2020). Tight compression: compressing CNN model tightly through unstructured pruning and simulated annealing based permutation. *DAC*.

[15] Liao, Z., Quétu, V., Nguyen, V. T., & Tartaglione, E. (2023). Can unstructured pruning reduce the depth in deep neural networks? *ICCV Workshops*, 1402–1406.

[16] Yang, Z., & Zhang, H. (2021). Comparative analysis of structured pruning and unstructured pruning. *International Conference on Frontier Computing*, 882–889.

[17] Kim, Y. D., Park, E., Yoo, S., Choi, T., Yang, L., & Shin, D. (2015). Compression of deep convolutional neural networks for fast and low power mobile applications. arXiv:1511.06530.

[18] Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017). Pruning filters for efficient ConvNets. *ICLR*.

[19] Ding, X., Ding, G., Han, J., & Tang, S. (2018). Auto-balanced filter pruning for efficient convolutional neural networks. *AAAI*, 32(1).

[20] Lin, S., Ji, R., Li, Y., Wu, Y., Huang, F., & Zhang, B. (2018). Accelerating convolutional networks via global & dynamic filter pruning. *IJCAI*.

[21] Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv:1803.03635.

[22] Molchanov, P., Mallya, A., Tyree, S., Frosio, I., & Kautz, J. (2019). Importance estimation for neural network pruning. *CVPR*, 11264–11272.

[23] You, Z., Yan, K., Ye, J., Ma, M., & Wang, P. (2019). Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks. *NeurIPS*, 32.

[24] Lee, J., Park, S., Mo, S., Ahn, S., & Shin, J. (2020). Layer-adaptive sparsity for the magnitude-based pruning. arXiv:2010.07611.

[25] Tang, Y., Wang, Y., Xu, Y., Deng, Y., Xu, C., Tao, D., & Xu, C. (2021). Manifold regularized dynamic network pruning. *CVPR*, 5018–5028.

[26] Tofigh, S., Ahmad, M. O., & Swamy, M. N. S. (2022). A low-complexity modified ThiNet algorithm for pruning convolutional neural networks. *IEEE Signal Processing Letters*, 29, 1012–1016.

[27] Park, S., Lee, J., Mo, S., & Shin, J. (2020). Lookahead: A far-sighted alternative of magnitude-based pruning. arXiv:2002.04809.

[28] Balderas, L., Lastra, M., & Benítez, J. M. (2023). Optimizing convolutional neural network architecture. arXiv:2401.01361.

[29] Li, G., Wang, J., Shen, H. W., Chen, K., Shan, G., & Lu, Z. (2020). CNNPruner: Pruning convolutional neural networks with visual analytics. *IEEE TVCG*, 27(2), 1364–1373.

[30] Fernandes Jr, F. E., & Yen, G. G. (2021). Pruning deep convolutional neural networks architectures with evolution strategy. *Information Sciences*, 552, 29–47.

[31] Ferreira, G. B., de Barros, A., Ibrahim, I., & Silva, R. (2023). Surrogate-based constrained multi-objective optimization for the compression of CNNs. *ENIAC*.

[32] Chang, J., Lu, Y., Xue, P., Xu, Y., & Wei, Z. (2022). Automatic channel pruning via clustering and swarm intelligence optimization for CNN. *Applied Intelligence*, 52(15), 17751–17771.

[33] Yu, S., Mazaheri, A., & Jannesari, A. (2021). Auto graph encoder-decoder for neural network pruning. *ICCV*, 6362–6372.

[34] Amelio, A., Bonifazi, G., Cauteruccio, F., Corradini, E., Marchetti, M., Ursino, D., & Virgili, L. (2023). Representation and compression of Residual Neural Networks through a multilayer network based approach. *Expert Systems with Applications*, 215, 119391.

[35] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … & Polosukhin, I. (2017). Attention is all you need. *NeurIPS*, 30.

[36] Xu, D., Yen, I. E., Zhao, J., & Xiao, Z. (2021). Rethinking network pruning — under the pre-train and fine-tune paradigm. arXiv:2104.08682.

[37] Yang, H., Liang, Y., Liu, W., & Meng, F. (2023). Filter pruning via attention consistency on feature maps. *Applied Sciences*, 13(3), 1964.

[38] Sun, H., Zhang, S., Tian, X., & Zou, Y. (2024). Pruning DETR: efficient end-to-end object detection with sparse structured pruning. *Signal, Image and Video Processing*, 18(1), 129–135.

[39] Wang, Y., Guo, S., Guo, J., Zhang, J., Zhang, W., Yan, C., & Zhang, Y. (2024). Towards performance-maximizing neural network pruning via global channel attention. *Neural Networks*, 171, 104–113.

[40] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *CVPR*, 11976–11986.

[41] Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2023). A simple and effective pruning approach for large language models. arXiv:2306.11695.

[42] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., … & Houlsby, N. (2020). An image is worth 16×16 words: Transformers for image recognition at scale. arXiv:2010.11929.

[43] Kuznedelev, D., Kurtic, E., Frantar, E., & Alistarh, D. (2024). CAP: Correlation-aware pruning for highly-accurate sparse vision models. *NeurIPS*, 36.

[44] He, H., Cai, J., Liu, J., Pan, Z., Zhang, J., Tao, D., & Zhuang, B. (2024). Pruning self-attentions into convolutional layers in single path. *IEEE TPAMI*.

[45] Guo, M. H., Lu, C. Z., Liu, Z. N., Cheng, M. M., & Hu, S. M. (2023). Visual attention network. *Computational Visual Media*, 9(4), 733–752.

[46] Kaelbling, L. P., Littman, M. L., & Moore, A. W. (1996). Reinforcement learning: A survey. *JAIR*, 4, 237–285.

[47] Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. *IJRR*, 32(11), 1238–1274.

[48] Kaiser, Ł., Babaeizadeh, M., Miłoś, P., Osiński, B., Campbell, R. H., Czechowski, K., … & Michalewski, H. (2019). Model-based reinforcement learning for Atari. *ICLR*.

[49] Mammeri, Z. (2019). Reinforcement learning based routing in networks: Review and classification of approaches. *IEEE Access*, 7, 55916–55950.

[50] Li, J., Monroe, W., Ritter, A., Galley, M., Gao, J., & Jurafsky, D. (2016). Deep reinforcement learning for dialogue generation. arXiv:1606.01541.

[51] Wiering, M. A., & Van Otterlo, M. (2012). Reinforcement learning. *Adaptation, Learning, and Optimization*, 12.

[52] Arulkumaran, K., Deisenroth, M. P., Brundage, M., & Bharath, A. A. (2017). Deep reinforcement learning: A brief survey. *IEEE Signal Processing Magazine*, 34(6), 26–38.

[53] Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256.

[54] Zoph, B., & Le, Q. V. (2016). Neural architecture search with reinforcement learning. arXiv:1611.01578.

[55] Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). Learning transferable architectures for scalable image recognition. *CVPR*, 8697–8710.

[56] Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., & Le, Q. V. (2019). MnasNet: Platform-aware neural architecture search for mobile. *CVPR*, 2820–2828.

[57] Wang, Z., & Li, C. (2022). Channel pruning via lookahead search guided reinforcement learning. *WACV*, 2029–2040.

[58] Pham, H., Guan, M., Zoph, B., Le, Q., & Dean, J. (2018). Efficient neural architecture search via parameters sharing. *ICML*, 4095–4104.

[59] Yang, Z., Wang, Y., Chen, X., Shi, B., Xu, C., Xu, C., … & Xu, C. (2020). CARS: Continuous evolution for efficient neural architecture search. *CVPR*, 1829–1838.

[60] Liu, C., Zoph, B., Neumann, M., Shlens, J., Hua, W., Li, L. J., … & Murphy, K. (2018). Progressive neural architecture search. *ECCV*, 19–34.

[61] Liu, H., Simonyan, K., & Yang, Y. (2018). DARTS: Differentiable architecture search. arXiv:1806.09055.

[62] Lopes, V., Carlucci, F. M., Esperança, P. M., Singh, M., Yang, A., Gabillon, V., … & Wang, J. (2023). MANAS: Multi-agent neural architecture search. *Machine Learning*.

[63] Dai, X., Chen, D., Liu, M., Chen, Y., & Yuan, L. (2020). DA-NAS: Data adapted pruning for efficient neural architecture search. *ECCV*, 584–600.

[64] Ding, Y., Wu, Y., Huang, C., Tang, S., Wu, F., Yang, Y., … & Zhuang, Y. (2022). NAP: Neural architecture search with pruning. *Neurocomputing*, 477, 85–95.

[65] Li, Y., Zhao, P., Yuan, G., Lin, X., Wang, Y., & Chen, X. (2022). Pruning-as-search: Efficient neural architecture search via channel pruning and structural reparameterization. arXiv:2206.01198.

[66] Dong, X., & Yang, Y. (2019). Network pruning via transformable architecture search. *NeurIPS*, 32.

[67] Wei, X., Zhang, N., Liu, W., & Chen, H. (2022). NAS-based CNN channel pruning for remote sensing scene classification. *IEEE GRSL*, 19, 1–5.

[68] Lee, S., & Song, B. C. (2023). Fast filter pruning via coarse-to-fine neural architecture search and contrastive knowledge transfer. *IEEE TNNLS*.

[69] Tanaka, H., Kunin, D., Yamins, D. L., & Ganguli, S. (2020). Pruning neural networks without any data by iteratively conserving synaptic flow. *NeurIPS*, 33, 6377–6389.

[70] Mellor, J., Turner, J., Storkey, A., & Crowley, E. J. (2021). Neural architecture search without training. *ICML*, 7588–7598.

[71] Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*, 248–255.

[72] Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.

[73] Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.

[74] Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A. Y. (2011). Reading digits in natural images with unsupervised feature learning. *NIPS Workshop on Deep Learning and Unsupervised Feature Learning*.

[75] Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A. (2017). Places: A 10 million image database for scene recognition. *IEEE TPAMI*, 40(6), 1452–1464.

[76] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv:1409.1556.

[77] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., … & Rabinovich, A. (2015). Going deeper with convolutions. *CVPR*, 1–9.

[78] Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., … & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv:1704.04861.

[79] He, Y., Lin, J., Liu, Z., Wang, H., Li, L. J., & Han, S. (2018). AMC: AutoML for model compression and acceleration on mobile devices. *ECCV*, 784–800. arXiv:1802.03494.

[80] Liu, Z., Mu, H., Zhang, X., Guo, Z., Yang, X., Cheng, K. T., & Sun, J. (2019). MetaPruning: Meta learning for automatic channel pruning. *ICCV*.

[81] Wang, X., Rachwan, J., Günnemann, S., & Charpentier, B. (2024). Structurally Prune Anything: Any architecture, any framework, any time. arXiv:2403.18955.

[82] Wu, X., Gao, S., Zhang, Z., Li, Z., Bao, R., Zhang, Y., Wang, X., & Huang, H. (2024). Auto-Train-Once: Controller network guided automatic network pruning from scratch. *CVPR*. https://doi.org/10.1109/cvpr52733.2024.01530

[83] Chen, T., Qu, X., Aponte, D., Banbury, C., Ko, J., Ding, T., Ma, Y., Lyapunov, V., Zharkov, I., & Liang, L. (2024). HESSO: Towards automatic efficient and user friendly any neural network training and pruning. arXiv:2409.09085.

[84] Tang, M., Liu, N., Yang, T., Fang, H., Lin, Q., Tan, Y., Chen, X., Liu, D., Zhong, K., & Ren, A. (2024). FreePrune: An automatic pruning framework across various granularities based on training-free evaluation. *IEEE TCAD*, 43(11), 4033–4044. https://doi.org/10.1109/tcad.2024.3443694

[85] Hu, Y., Chen, Y., Zou, X., & Liu, Y. (2025). Automatic channel pruning by neural network based on improved poplar optimisation. *Knowledge-Based Systems*, 310, 113002. https://doi.org/10.1016/j.knosys.2025.113002

[86] Palakonda, V., Tursunboev, J., Kang, J. M., & Moon, S. (2025). Metaheuristics for pruning convolutional neural networks: A comparative study. *Expert Systems with Applications*, 268, 126326. https://doi.org/10.1016/j.eswa.2024.126326

[87] Chen, S., & Zhao, Y. (2025). MLPruner: Pruning convolutional neural networks with automatic mask learning. *PeerJ Computer Science*, 11, e3132. https://doi.org/10.7717/peerj-cs.3132

[88] Chahbouni, A., El Manaa, K., Abouch, Y., El Manaa, I., Bossoufi, B., El Ghzaoui, M., & El Alami, R. (2025). Attention-guided differentiable channel pruning for efficient deep networks. *Machine Learning and Knowledge Extraction*, 7(4), 110. https://doi.org/10.3390/make7040110

[89] Pham, V. T., Zniyed, Y., & Nguyen, T. P. (2025). Singular values-driven automated filter pruning. *Neural Networks*, 192, 107857. https://doi.org/10.1016/j.neunet.2025.107857

[90] Samarin, A., Nazarenko, A., Kotenko, E., Toropov, A., Savelev, A., Motyko, A., & Malykh, V. (2026). Flow-guided neural pruning: Signal-flow framework for multi-architecture model compression. *Machine Learning and Knowledge Extraction*, 8(8), 236. https://doi.org/10.3390/make8080236

[91] Zhou, G., & Zhang, D. (2026). A dependency-aware global spectral-entropy framework for structured neural network pruning. *Applied Soft Computing*, 116186. https://doi.org/10.1016/j.asoc.2026.116186

[92] Fang, Y.-C., Li, W.-Z., Zeng, Y., Lu, Q.-N., & Lu, S.-L. (2025). Pushing to the limit: An attention-based dual-prune approach for highly-compacted CNN filter pruning. *Journal of Computer Science and Technology*, 40(3), 805–820. https://doi.org/10.1007/s11390-024-3536-3

[93] Ghimire, D., Kil, D., Jeong, S., Park, J., & Kim, S.-h. (2026). One-cycle structured pruning via stability-driven subnetwork search. *WACV*. arXiv:2501.13439. https://github.com/ghimiredhikura/OCSPruner

[94] Liu, Z., Cao, Y., Yu, Y., Qi, H., & Gui, J. (2025). Structure-aware automatic channel pruning by searching with graph embedding. arXiv:2506.11469.

[95] Li, X., & Xiu, X. (2025). GoPrune: Accelerated structured pruning with ℓ2,p-norm optimization. arXiv:2511.22120. https://github.com/xianchaoxiu/GoPrune

[96] Liu, X., Li, M., Li, X., Qu, L., Wang, G., Peng, Z., Song, Y., Liu, Z., Jiang, L., & Li, J. (2025). Enhanced structured lasso pruning with class-wise information. arXiv:2502.09125.
