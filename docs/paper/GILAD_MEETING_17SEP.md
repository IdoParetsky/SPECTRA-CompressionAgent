# SPECTRA — meeting brief for Dr. Gilad Katz (17 Sep 2026)

**From:** Ido Paretsky  
**For:** meeting 17 September 2026  
**As of:** 17 Sep 08:55 IDT (overnight trains still running)  
**Merges:** the 16 Sep executive note (university postponement) into this meeting brief.

**Status 18 Sep 01:12 (Ido):** Gilad granted a **full-semester thesis extension**. The clocks below are the *ask as of 17 Sep*, not the live constraint. Do not optimize remaining work for 30 Sep / 30 Oct / mid-December. Do not invent a new university date. Rank remaining cells for **purity of science**. This file stays as the meeting record.

**Two clocks** *(historical ask — superseded as a kill switch)*

| Clock | Date | What it is for |
|---|---|---|
| Working | **30 October 2026** (one month from 30 Sep) | Last train finishes, TESTs on held-out catalogs, first complete draft. |
| University | **mid-December 2026** (~2½ months from 30 Sep) | Formal postponement I am asking you to support: comments, polish, and slack if TESTs run long. Not a new method. |

Numbers below are TEST trajectory points (`TRAJ val_best`: most compressed in-band point on **validation**; test Δacc reported). They are PRELIM unless locked in the ledger. We do not pick the quoted point on the test set.

---

## Bottom line

Path 3 showed that the old frozen agent was not a learned schedule. v2 showed that the optimiser can learn one — a peaked, non-uniform policy, for the first time. v3/V4 are the first training loop that can *keep* learning after the first lucky freeze, and that can choose *which* filters die — not only how many. Those last-train jobs are still running (~episode 45–59 of a 250-episode floor). We do not yet have a TEST of v3 or V4. Overnight, the probe scores that pick TESTable snapshots **did not beat their first freeze**. The 2-pass mild control is now in: extra cut on the easy net, same keep on the hard net. We still do not beat same-loop heuristics at equal size as a family claim.

The scientific hole that remains is **generalisation of that policy**, not “does reinforcement learning run.” A 30 September submission can honestly tell the story as a **methods + diagnostic thesis**, with v2b as the peaked-policy existence proof and Path 3 as the protocol correction. It cannot honestly present v3/V4 as the thesis agent, nor finish the coverage matrix you asked for (family × dataset) plus a NEON-style Pareto with the new actors on it. It would freeze a half-finished product just as the method became a real learned agent.

I ask you to hold **30 October** as the working date (last train + TESTs + a first complete draft), and to support a university postponement to **mid-December**, so writing and committee process are not squeezed if the experiment card slips. The claim is the one you set on 18 August: a **frozen generic DRL pruner** that transfers, competitive enough versus heuristics, not a fight with DepGraph on its home cell.

---

## The claim (unchanged)

SPECTRA is NEON moved to CNNs: train one agent offline on many architectures **and** datasets; freeze it; prune an unseen CNN with no per-target agent training. Gilad’s bar (18 Aug): competitive-enough while transferring. Keep **both** artifacts — coverage matrix (family × dataset) and a NEON-style Pareto (Δacc vs size vs heuristics and quoted literature). Do not claim to beat focused SOTA on their home architecture × dataset. No ImageNet DRL train. **C100 belongs in the train pool** (recoverable nets first); ImageNet is the held-out dataset probe. A C10-only agent measured on C100 (ledger §21) is **not** the thesis transfer cell — Ido 17 Sep 14:45.

**Expensive mix-up (do not).** Changing the optimiser, the reward it is scored on, and the train catalog in one job is uninterpretable. Same family of overlays that failed to move skinny r56-w4. A frozen-snap TEST of the current loop lands first; then V5 changes **one** of those three.

---

## Path 3 → v2 → v3 → v4 (what was missing, what the next batch fixed)

### Path 3 — honest TEST of the old frozen 10-net actor

**What we had.** Offline agent, L1 ranking inside the environment, NEON-style reward, three rates `{1.0, 0.9, 0.8}`.

**What was lacking**

- TEST **sampled** the policy and never put the actor/critic in `.eval()`, so encoder dropout was live. The same actor on the same net could land ~10 pp apart in kept size. Seed spreads of 1–3 pp were inside that noise.
- The trained policy was **uniform**. Argmax was the head’s bias, which is always the gentlest legal cut (rate 0.9) — i.e. the “mild” heuristic. There was no learned schedule to transfer.
- Training episodes were **5 steps**. TEST walks tens of layers. Most of the net never got a policy gradient.
- Residual **groups** were cut on every owning row. One 0.8 on a skinny ResNet-56 stream is not a 20% size cut; it is a cliff.
- Prefer/cubes “DRL” arms **bypassed the actor**. They were a deterministic heuristic, reported as if they were three-seed DRL.

**What Path 3 fixed (protocol, not a new agent)**

- Replay the frozen actor with **argmax** (`det=1`) and a **trajectory** protocol: walk the net, pick the most compressed point whose **val** Δacc stays inside τ = 10 pp, report **test** Δacc.
- Result (thin CIFAR-10): argmax ≡ mild on the easy net; on ResNet-56-w4 the policy walks to the greedy cliff (**−25.2 @ 0.667**), not the kinder sampled locked row (**−15.9 @ 0.704**). The locked number was a lucky sample.
- Ranking A/B (same actor, only the filter sorter changes): FPGM slightly kinder than L1 on the cliff, BN-scale worse. **Keep L1** as the default sorter until the *agent* is allowed to choose ranking.

**One-liner.** Path 3 did not invent a better agent. It stopped us from quoting a uniform policy as DRL.

---

### v2 — first peaked policy (13–15 Sep)

**What we changed.** PPO; cold zero-init head (no leftover bias toward 0.9); **group-once** (cut each coupled stream at most once per pass); state channels for slack / progress / kept ratio; reward on the **realised** cut with cube-root scaling (`structural` + `cbrt`); TEST is always argmax TRAJ.

**What was lacking in Path 3 that v2 fixed**

- Optimiser never left uniform → v2a/v2b left uniform by ~episode 8 (`gap_to_uniform` +0.19 to +0.40). The critic’s explained variance went positive. That is “RL is running.”
- 5-step episodes / no group-once → full walks + group-once. The r20 plateau moved from Path 3’s ~0.48 keep (plain walk) to **0.746** (one cut per group). That lever is the environment, not magic in the actor.
- Hidden L1-only ranking → **v2b** is a 5-button menu: identity, or (keep 90% or 80%) × (L1 or FPGM). The agent can choose *which* filters die.

**What we TESTed (v2b episode-15 snapshot)**

- Thin ResNet-20: **−1.2 @ 0.606**, 1.9 pp kinder than L1-once at the same keep (shy of a 2 pp win). First frozen-actor TEST off the mild plateau.
- Thin ResNet-56: **−7.1 @ 0.879** at equal Δacc and **less keep** than mild-once (0.923).
- Unlike family (ShuffleNet / RepVGG): same keep as L1-once; L1 is kinder on RepVGG. **Not** a ranking-transfer win.
- Similar family: in-band on VGG, MobileNet, DenseNet-100 (**−2.5 @ 0.662**). Hard cell remains skinny ResNet-56-w10 (**−6.1 @ 0.642**, val tight on τ).
- CIFAR-100 of a **C10-only** v2b actor: all five nets **identity**. That is not a C10→C100 success, and C10→C100 was **never** the intended claim (Ido 17 Sep). We will not overwrite the §21 C10-only-actor table with this catalog. Next train batch puts recoverable C100 **into** train.

**What was still lacking (why v2 is not the thesis agent)**

- **v2a** (3 rates, L1 only) **cloned mild** on every TESTed snap, including the train-best snapshot. Raising the train score did not buy a new walk.
- **v2c** (original NEON reward, unscaled) collapsed: return scale in the thousands, critic dead. We do not TEST it as DRL.
- Training **stopped for the wrong reason**: a noisy 4-net train score (`batch_score`) failed to beat a lucky max. v2b’s best freeze is still episode 15 of 116; later episodes were still mixed, not collapsed. That is a governor failure, not “learning finished.”
- The policy almost never saw the τ-band **edge** in training (over-budget on ~1–3% of train steps). At TEST, skinny nets spend τ in 8–12% of cut and the actor keeps cutting.
- No **group cost** in the state, so “this layer is cheap / expensive because of its residual friends” was invisible.
- Beating heuristics at **equal keep** is still a must. We have not met it as a family claim. Cheap abort (skip the last train) is **off**.

**One-liner.** v2 proved the student can learn. It froze the first good homework too early and never showed the student the hard exam questions.

---

### v3 — last train, still running (from 16 Sep 09:55)

**What we changed (fixes aimed at v2’s holes)**

- **Governor:** freeze a snapshot when a deterministic argmax walk on two probe nets improves. Training does **not** stop. Patience and rewind look at that probe, not at a lucky 4-net train max.
- **Band edge:** two compression **passes** in training (TEST replays two). The actor sees “slack is gone, stop.”
- **Group cost** in the state (parameter/MAC share of the whole coupled group).
- **Lifetime:** at least 250 episodes; patience 150 on the probe; rewind to the elite up to 3 times.
- **Catalog:** 24 training nets (widths that sit between the old train set and the skinny hold-outs). No overlap with any TEST catalog.
- **Four sibling menus**, each 5 buttons `identity | (0.9/0.8) × {L1, RANK}`: RANK = FPGM, SVD, or BN-scale, all with scaled reward; plus FPGM × **original NEON reward, unscaled** (the fair “was it the menu or the scale” cell that v2c never gave us on this menu).

**What we know from training telemetry (not TESTs)** — as of 08:55, ~23 h in

- Critics are alive on the scaled-reward arms. The unscaled NEON arm survived the collapse rule we used to kill v2c, then **started cutting** at the second probe — and at the fourth probe went back to identity.
- First frozen snaps already prune on the probe nets, so they are TESTable. None of the three scaled ranking arms has **beaten its first probe**. The six-probe window (~episode 70) is still open. We are **not** stopping them because the calendar says 17 Sep.
- Throughput is slower than v2b (~2 episodes/h vs ~5.6). After a night, v3 is at episode ~45–52; v2b would already have been past 100. The governor is doing its job: training continues after the first freeze, which is exactly what killed v2b at episode 15 of 116.

**What is still missing until TEST lands.** Whether any of this walks held-out nets better than **2-pass** group-once L1/mild — the fair control, because v3 trains with two passes.

**One-liner.** v3 is v2 with a teacher who does not send the student home after the first good quiz, and who finally shows the cost of cutting a residual stream.

---

### V4 — two heads: how much, then which (from 16 Sep 12:04)

**What was lacking in v3.** A 5-button *joint* menu still ties “how much” to “which sorter” as one combo. Growing to five ranking rules as 13–15 joint buttons dilutes credit (L1/L2/SVD are almost the same sorter).

**What V4 does.** One shared encoder; two lists. Head 1: keep 100 / 90 / 80%. Head 2: L1, FPGM, BN-scale, SVD, or Taylor. Ranking is **off** on identity (no pretend choice when nothing is cut). Taylor is the only sorter that looks at a batch of data (`|w · gradient|`). L2 is out (too correlated with L1).

**Status.** Running, episode ~59, PPO-15. Probe rose 0.115 → 0.210 at the second freeze, then held 0.210 at probes 3 and 4. TESTable snap is still episode 23. Critic dipped negative at PPO-13 and recovered. No TEST yet.

**One-liner.** V4 is the first agent that can say “cut 20%” and “use FPGM” as two thoughts, not as one pre-baked pair.

---

## Overnight (16–17 Sep) — telemetry, not TEST

v2b left uniform by episode 8, froze a TESTable snap at episode 15, and then **died** at 116 because a noisy train score never beat that max. v3/V4 were built so that freeze does not stop the job. After one night that loop is working — and the probe that is supposed to *improve* has not yet.

Probe score = mean compression (`1 − kept`) of an argmax walk on two skinny probe nets. Higher is more compressed. A new snapshot freezes only when this rises.

| Arm | Probe 1 (ep 12) | 2 (ep 24) | 3 (ep 36) | 4 (ep 48) | TESTable snap |
|---|---|---|---|---|---|
| v3 FPGM | **0.262** | 0.000 identity | 0.241 | (walking) | ep 11 |
| v3 SVD | **0.262** | 0.210 | 0.210 | (walking) | ep 11 |
| v3 BN-scale | **0.210** | 0.210 | 0.202 | (walking) | ep 11 |
| v3 FPGM × NEON-raw | 0.000 identity | **0.262** | 0.262 | 0.000 identity | ep 23 |
| V4 factored | 0.115 | **0.210** | 0.210 | 0.210 | ep 23 |

**Good.** Critics on the scaled-reward arms stay healthy. Training did not stop at the first freeze (the v2b failure). The unscaled NEON cell did not die the way v2c did. 2-pass mild control is now a TEST: easy net **−3.4 @ 0.536**, hard net **−6.6 @ 0.923** (same keep as 1-pass mild).

**Bad.** No later probe has beaten the first freeze on the three scaled ranking arms. NEON-raw’s argmax went back to “do nothing” at probe 4. V4 is stuck at the mild-plateau keep on the skinny probe net. Throughput is ~3× slower than v2b, so the 250-episode floor is still days away. **No v3/V4 TEST yet** — QOS is full with trains.

---

## Mechanisms, in one place

**Sampling vs argmax.** Old TESTs drew a random action from the policy (and dropout was on). Path 3 onward: **argmax**, actor in eval mode. For a peaked policy those two disagree; for a uniform policy argmax is just the bias (mild). Thesis tables use argmax.

**How we pick the quoted number.** Walk the TEST net. At every step, record val Δacc and size. Quote the most compressed point that is still inside τ on **val**. Report its **test** Δacc. Never shop on the test set. Skip wrap means, train-loader accuracy, and terminals whose val already left τ.

**Reward.** NEON’s shape is kept: in-band credit, over-band penalty, optional accuracy-gain bonus. v2a/v2b/v3-cbrt/V4 score the **realised** parameter/FLOP cut, cube-root scaled, so a 20% cut on a 4-channel stem is not the same bonus as a 20% cut on a wide residual stream. v2c and v3-neonraw use the original **nominal-rate** NEON formula, unscaled. Scaling is what made the critic learn; it does not prove the trichotomy is wrong. You have not yet replied to the reward-function questions I sent. I would still value your judgement before we lock thesis prose.

**Which filters.** Default sorter is L1 (Li et al., ICLR 2017). v2b/v3 let the agent pick L1 vs one partner (FPGM / SVD / BN-scale). V4 picks among five, including Taylor (Molchanov et al.). Heuristics use the same environment, so “beats greedy” is a **schedule** comparison unless the menu itself chooses ranking.

**Group-once.** Residual/DenseNet streams share width. Cutting every owner is a cliff. One cut per group per pass is the lever that moved Path 3’s r20 ~0.48 keep to the 0.746 plateau. v2+ trains and TESTs with this on.

**Governor.** v2 froze on a lucky 4-net train score and died on patience. v3/V4 freeze a copy when a two-net argmax probe improves, and keep training (rewind to that elite if the probe goes stale).

---

## Where we are this afternoon (17 Sep 14:50)

| Piece | State |
|---|---|
| v2b TESTs | Thin, unlike, similar (skip r32), C100 — in. Mild similar DenseNet still walking (step ~91). L1 similar DenseNet was sacrificed for GPU. |
| 2-pass heuristic controls | Both thin **in**. Mild §93: r20 **−3.4 @ 0.536**, r56 **−6.6 @ 0.923**. L1 §94: r20 **−7.3 @ 0.417**, r56 **−7.8 @ 0.898**. |
| First v3 thin TRAJ | **21428727 COMPLETED §95.** fpgm ep0011: r20 **−5.1 @ 0.536**, r56 **−6.8 @ 0.923**. Equal keep vs 2-pass mild; 1.7 / 0.2 pp worse. Cloned mild keep. svd TRAJ **21433272** PD. |
| v3 (4 arms) + V4 | Training. V4 6th probe **0.241** (first arm to beat its first freeze). Calendar freeze is **symbolic**. |
| ImageNet DRL | Not started. Frozen C10/**C100-trained** → ImageNet is the later dataset probe. |
| Gilad win vs heuristics | **Not met** as a family claim. Cheap abort off. |

### Oral follow-up from the meeting (17 Sep ~15:21) — for the record

**1. “Do I retrain the layer after pruning?”** Three recipes, not one. (a) What we run today: full-net Adam FT, remaining filters **kept**. (b) What I first answered: freeze the net, FT the pruned layer’s **kept** weights — SPECTRA’s unused flag; **0/32 OK** on ResNet-20/56. (c) What you reminded me NEON did: freeze the net, **throw away** the remaining filters of the edited layer, reinit, train **until convergence**. SPECTRA’s code currently aliases that path to ordinary prune-and-keep. We will implement (c), train an agent that way, and TEST it against the current best. Implementation is a Fable sitting; we will not overlay the live v3/V4 trains.

**2. Evaluation must follow CNN-pruning convention.** After merit, the committee will ask for SPECTRA next to SOTA on **their** nets. We will TEST the frozen agent **and** same-loop heuristics on literature-home cells (at minimum CIFAR-10 ResNet-56 at ~94% origin — not thin-w4), and print `origin | pruned | Δacc | params kept | FLOPs kept`. Quote DepGraph / OCS / FPGM / HRank / Slimming / ATO with origin and FT captions. Do not claim to beat focused SOTA on their home cell.

Draft algorithms of every stage (Path 3 → V5) are in `docs/paper/LOOP_ALGORITHMS.md` for markup.

---

## 30 September — is a full TEST catalog feasible?

**If “full catalog” means every v3/V4 arm × thin + similar + unlike + CIFAR-100:** **no.** One similar catalog is ~2 days on one GPU. Five actors × four catalogs is more GPU-time than 13 days with five GPUs still in training. Trains themselves have a 250-episode floor (~several more days at current speed).

**If “full catalog” means: finish the v2b controls, run 2-pass heuristics, and TEST the frozen v3/V4 snapshots on the two thin nets (and maybe unlike):** **yes, if GPUs keep moving.** Both 2-pass heuristics are in. First v3 thin TRAJ is queued. Similar-family for v3/V4 by the 30th is a stretch, not a plan. Trains are ~50 of 250 episodes — several more days.

### What we can still aim to submit on 30 Sep (if thin TESTs of v3/V4 land)

A defensible **methods thesis**, not a finished transfer thesis:

- The protocol is honest (argmax, val-selected, group-once, quoted TEST only).
- Path 3: old frozen agent ≡ mild / greedy-cliff; sampling had been inflating the story.
- v2b: first peaked generic policy; in-band on similar CIFAR-10 families; unlike ≡ L1 keep; C100 identity.
- v3/V4: specified last train; thin TESTs if they complete; captioned as PRELIM if they do not beat 2-pass heuristics.
- Coverage matrix: C10 similar/unlike from v2b; C100 of a C10-only actor is a **measurement**, not the transfer claim. Next agent trains on diverse families × datasets. Pareto: v2b vs mild vs L1 vs Path 3, plus literature stars **quoted**, not re-run.
- Claim we can stand behind: *we built a transferable DRL pruning loop that actually learns, and we measured it fairly.* Claim we cannot stand behind: *the last agent beats heuristics on held-out families and ranking is why.*

### What we give up if we abide 30 Sep

- v3/V4 as the thesis agent (training unfinished; most hold-out TESTs missing).
- The ranking-as-action claim (v2b already looks like L1 on unlike; v3/V4 TESTs are the test of that sentence).
- Two-pass fair controls on the similar/unlike catalogs.
- Time to write, take comments, and put both the coverage matrix and the Pareto in the draft at thesis quality.
- Any ImageNet transfer probe.
- A second look if v3’s probe stays stuck — rewind and later snaps would not make the tables.

---

## What another month (to 30 October) is for

Not “more ideas.” Finish the loop we already specified:

1. Let v3/V4 reach their designed lifetime (or rewind/patience), TEST frozen snaps on **thin**, then **similar**, then **unlike**, CIFAR-100 last. Fair controls = 2-pass group-once mild and L1.
2. Close G1: mild (and, if you want the trio, L1) similar DenseNet, so v2b vs heuristics is a complete family table.
3. If two or more last-train arms have thin TESTs: a **validation-chosen portfolio** (per net, pick the actor with the best val point; report its test Δacc). Caption: not a weight merge.
4. Write the thesis around the 18 Aug frame: coverage + Pareto, transfer, no home-SOTA boast. Lock reward-function prose after this meeting.
5. Optional if GPUs free: frozen C10 agent on ImageNet as a short transfer probe — still **no** ImageNet DRL train.

That is one month of TEST + write, not a new research programme.

**Mid-December** is the university-side clock: time to take comments, polish both the coverage matrix and the Pareto to thesis quality, and absorb slip if v3/V4 TESTs run long. Two and a half months is enough to finish the campaign we already specified and write a complete thesis. It is not a second research programme.

---

## Why I am asking (justifications, short)

1. **The scientific object changed in the last two weeks.** Before Path 3 we would have submitted a uniform policy as a generic DRL agent. That would have been a wrong thesis.
2. **The optimiser works; the transfer tables for the *right* agent are not in yet.** Submitting now documents a diagnosis plus v2b, and leaves the last train unTESTed.
3. **Gilad’s own win condition is equal-keep vs heuristics.** We have not met it. The last train is the attempt. Killing it for a calendar freeze guarantees we never measure it.
4. **The last train is already specified and running.** An extra month is evaluation and writing, not a new method hunt (no new reward, no ImageNet DRL, no encoder restart).
5. **Two clocks, one programme.** 30 October is enough for TESTs + a first complete draft. Mid-December is the university postponement (~2½ months) so comments and polish are not squeezed. The tables today are not yet the thesis you would want to sign; they will not become that by 30 September either.

---

## The ask

Please support **both** clocks:

1. Hold **30 October 2026** as the **working** thesis date (experiment card + first complete draft).
2. Support a request to the university to postpone submission by about **two and a half months**, to **mid-December 2026**.

The reason is not delay for its own sake. We now have a peaked policy, a last train that is specified and already running, and a TEST protocol that does not cheat on the test set. 30 October is the experiment. Mid-December is writing and process.

I remain available to discuss the reward function whenever you wish. The last train proceeds with the scaled reward as the default and the original NEON reward as the explicit control, unless you prefer otherwise.

I will keep the 18 August frame, quote TEST only, and not claim a heuristic-family win until the tables show it. v3/V4 keep running until we TEST them or you tell me to stop.

Respectfully,  
Ido Paretsky

---
---

# SPECTRA — סיכום לישיבה עם ד״ר גלעד כץ (17 בספטמבר 2026)

**מאת:** עידו פרצקי  
**לישיבה:** 17 בספטמבר 2026  
**נכון ל:** 17 בספטמבר 08:55 (אימוני הלילה עדיין רצים)  
**מאחד:** את מכתב 16 בספטמבר (דחייה אוניברסיטאית) לתוך סיכום הישיבה הזה.

**שני שעונים**

| שעון | תאריך | בשביל מה |
|---|---|---|
| עבודה | **30 באוקטובר 2026** (חודש מ־30 בספטמבר) | האימון האחרון נגמר, TEST על הקטלוגים המוחזקים, טיוטה שלמה ראשונה. |
| אוניברסיטה | **אמצע דצמבר 2026** (~חודשיים וחצי מ־30 בספטמבר) | דחייה פורמלית שאני מבקש שתתמוך בה: הערות, ליטוש, וריווח אם ה־TESTs נמשכים. לא שיטה חדשה. |

המספרים למטה הם נקודות מסלול TEST (`TRAJ val_best`: הנקודה הדחוסה ביותר שעדיין בתוך הסף ב־**ולידציה**; מדווחים את שינוי הדיוק במבחן). הם ראשוניים אלא אם ננעלו בפנקס. לא בוחרים את הנקודה המצוטטת על סט המבחן.

---

## שורה תחתונה

Path 3 הראה שהסוכן הקפוא הישן לא היה לוח־זמנים שנלמד. v2 הראה שהאופטימייזר מסוגל ללמוד אחד — מדיניות לא־אחידה עם שיא, לראשונה. v3/V4 הם לולאת האימון הראשונה שיכולה *להמשיך* ללמוד אחרי הקפאה ברת־מזל ראשונה, ושבוחרת *אילו* מסננים מתים — לא רק כמה. עבודות האימון האחרונות עדיין רצות (~פרק 45–59 מרצפת 250). עדיין אין TEST של v3 או V4. בלילה, ציוני הבדיקה שבוחרים צילומים **לא עברו את ההקפאה הראשונה**. בקרת העדין הדו־מעבר כבר בפנים: חיתוך נוסף ברשת הקלה, אותה השארה ברשת הקשה. עדיין אין ניצחון על היוריסטיקות באותו לולאה באותו גודל כטענת משפחה.

החור המדעי שנותר הוא **הכללה של המדיניות הזו**, לא עצם השאלה אם למידת חיזוק רצה. הגשה ב־30 בספטמבר יכולה לספר את הסיפור הזה ביושר כ**תזת שיטה ואבחון**, עם v2b כהוכחת קיום למדיניות עם שיא ועם Path 3 כתיקון הפרוטוקול. היא לא יכולה להציג ביושר את v3/V4 כסוכן התזה, ולא להשלים את מטריצת הכיסוי שביקשת (משפחה × סט נתונים) ופארטו בסגנון NEON עם הסוכנים החדשים עליו. היא תקפיא מוצר חצי־אפוי בדיוק כשהשיטה הפכה לסוכן שנלמד באמת.

אני מבקש להחזיק את **30 באוקטובר** כתאריך העבודה (אימון אחרון + TEST + טיוטה שלמה ראשונה), ולתמוך בדחייה אוניברסיטאית ל־**אמצע דצמבר**, כדי שהכתיבה וההליך לא יידחקו אם כרטיס הניסוי מחליק. הטענה היא זו שקבעת ב־18 באוגוסט: **סוכן דילול מובנה גנרי קפוא** שעובר בין רשתות, תחרותי מספיק מול יוריסטיקות, לא קרב עם DepGraph על תא הבית שלו.

---

## הטענה (לא השתנתה)

SPECTRA הוא NEON שעבר ל־CNN: מאמנים סוכן אחד אופליין על ארכיטקטורות **וסטים** רבים; מקפיאים; מדללים CNN שלא נראה באימון בלי אימון סוכן ליעד. הרף של גלעד (18 באוגוסט): תחרותי מספיק תוך העברה. שומרים על **שני** התוצרים — מטריצת כיסוי ופארטו בסגנון NEON (שינוי דיוק מול גודל מול יוריסטיקות וספרות מצוטטת). לא טוענים שאנחנו מנצחים SOTA ממוקד על ארכיטקטורת הבית × הסט שלו. אין אימון DRL על ImageNet. **C100 שייך לברכת האימון** (רשתות ברות־החלמה קודם); ImageNet הוא מבחן הסט המוחזק. סוכן C10־בלבד שנמדד על C100 (פנקס §21) **אינו** תא ההעברה של התזה — עידו 17 בספטמבר.

**ערבוב יקר (לא).** שינוי האופטימייזר, הגמול שהוא נמדד עליו, וקטלוג האימון באותה עבודה הוא לא־ניתן־לפירוש. אותה משפחת שכבות שלא הזיזה את r56-w4 הרזה. קודם TEST של צילום קפוא מהלולאה הנוכחית; אחר כך V5 משנה **אחד** משלושתם.

---

## Path 3 → v2 → v3 → v4 (מה חסר, מה תיקן הסבב הבא)

### Path 3 — מבחן הוגן לסוכן 10־הרשתות הקפוא

**מה היה.** סוכן אופליין, דירוג L1 בתוך הסביבה, גמול בסגנון NEON, שלושה שיעורים `{1.0, 0.9, 0.8}`.

**מה חסר**

- ה־TEST **דגם** מהמדיניות ולא העביר את השחקן/מבקר ל־`.eval()`, כך שנשירת המקודד הייתה חיה. אותו סוכן על אותה רשת יכול היה ליפול כ־10 נקודות אחוז בשיעור ההשארה. פיזור „זרעים“ של 1–3 נקודות היה בתוך הרעש הזה.
- המדיניות המאומנת הייתה **אחידה**. הארגמקס היה הטיית הראש, כלומר תמיד החיתוך החוקי העדין ביותר (שיעור 0.9) — היוריסטיקה ה„עדינה“. לא היה לוח־זמנים שנלמד שאפשר להעביר.
- פרקי האימון היו **5 צעדים**. ה־TEST הולך על עשרות שכבות. רוב הרשת לא קיבלה גרדיאנט מדיניות.
- **קבוצות** שיוריות נחתכו בכל שורת בעלות. 0.8 על זרם של ResNet-56 צר אינו חיתוך של 20% מהגודל; זה צוק.
- זרועות prefer/cubes „DRL“ **עקפו את השחקן**. זו הייתה יוריסטיקה דטרמיניסטית שדווחה כאילו הייתה DRL בשלושה זרעים.

**מה Path 3 תיקן (פרוטוקול, לא סוכן חדש)**

- להריץ את הסוכן הקפוא ב־**ארגמקס** (`det=1`) ובפרוטוקול **מסלול**: הולכים על הרשת, בוחרים את הנקודה הדחוסה ביותר שבה ירידת ה־**ולידציה** נשארת בתוך τ = 10 נקודות אחוז, מדווחים את שינוי הדיוק ב־**מבחן**.
- תוצאה (CIFAR-10 דק): ארגמקס ≡ עדין על הרשת הקלה; על ResNet-56-w4 המדיניות הולכת לצוק החמדן (**−25.2 @ 0.667**), לא לשורה הנעולה העדינה יותר מהדגימה (**−15.9 @ 0.704**). המספר הנעול היה דגימה ברת־מזל.
- A/B דירוג (אותו סוכן, רק סדר המסננים משתנה): FPGM מעט עדין יותר מ־L1 על הצוק, סולם־BN גרוע יותר. **משאירים L1** כברירת מחדל עד שה־*סוכן* רשאי לבחור דירוג.

**משפט אחד.** Path 3 לא המציא סוכן טוב יותר. הוא הפסיק אותנו מלצטט מדיניות אחידה כ־DRL.

---

### v2 — מדיניות עם שיא, לראשונה (13–15 בספטמבר)

**מה שינינו.** PPO; ראש באתחול אפס (בלי הטיה שאריות ל־0.9); **חיתוך פעם אחת לקבוצה**; ערוצי מצב לריווח / התקדמות / שיעור השארה; גמול על החיתוך **שהתממש**, עם כיול שורש־שלישי; TEST תמיד ארגמקס־מסלול.

**מה היה חסר ב־Path 3 ו־v2 תיקן**

- האופטימייזר לא עזב אחידות → v2a/v2b עזבו אחידות בסביבות פרק 8. שונות מוסברת של המבקר הפכה חיובית. זה „למידת חיזוק רצה.“
- פרקי 5 צעדים / בלי חיתוך־פעם־לקבוצה → הליכות מלאות + חיתוך פעם לקבוצה. רמת ה־r20 זזה מ־~0.48 השארה ב־Path 3 ל־**0.746**. זה מנוף של הסביבה, לא קסם בשחקן.
- דירוג L1 סמוי → **v2b** הוא תפריט של 5 כפתורים: זהות, או (השאר 90% או 80%) × (L1 או FPGM). הסוכן יכול לבחור *אילו* מסננים מתים.

**מה בדקנו (צילום פרק 15 של v2b)**

- ResNet-20 דק: **−1.2 @ 0.606**, עדין ב־1.9 נקודות מ־L1־פעם באותה השארה (מתחת לרף 2 נקודות). TEST ראשון של סוכן קפוא מחוץ לרמת העדין.
- ResNet-56 דק: **−7.1 @ 0.879** באותו שינוי דיוק ו**פחות** השארה מ־עדין־פעם (0.923).
- משפחה שונה (ShuffleNet / RepVGG): אותה השארה כמו L1־פעם; L1 עדין יותר על RepVGG. **לא** ניצחון העברת־דירוג.
- משפחה דומה: בתוך הסף על VGG, MobileNet, DenseNet-100 (**−2.5 @ 0.662**). התא הקשה נשאר ResNet-56-w10 צר (**−6.1 @ 0.642**, ולידציה צמודה ל־τ).
- CIFAR-100 של סוכן v2b **C10־בלבד**: חמש הרשתות **זהות**. זה לא הצלחת C10→C100, ו־C10→C100 **מעולם לא הייתה** הטענה (עידו 17 בספטמבר). לא נדרוס את טבלת §21. סבב האימון הבא מכניס C100 בר־החלמה **לאימון**.

**מה עדיין חסר (למה v2 אינו סוכן התזה)**

- **v2a** (3 שיעורים, רק L1) **העתיק עדין** בכל צילום שנבדק, כולל הטוב באימון. העלאת ציון האימון לא קנתה הליכה חדשה.
- **v2c** (גמול NEON מקורי, בלי כיול) קרס: סקאלת תשואה באלפים, מבקר מת. לא בודקים אותו כ־DRL.
- האימון **נעצר מסיבה שגויה**: ציון אימון רועש על ארבע רשתות לא עבר מקסימום בר־מזל. הצילום הטוב של v2b הוא עדיין פרק 15 מתוך 116; הפרקים המאוחרים היו מעורבים, לא קרוסים. זה כשל מושל, לא „הלמידה נגמרה.“
- המדיניות כמעט לא ראתה את **קצה** סף־τ באימון. ב־TEST הרשתות הצרות שורפות את τ ב־8–12% חיתוך והשחקן ממשיך לחתוך.
- אין **עלות קבוצה** במצב, כך ש„השכבה הזו זולה/יקרה בגלל החברים השיוריים“ היה בלתי נראה.
- ניצחון על יוריסטיקות ב**אותה השארה** עדיין חובה. לא עמדנו בזה כטענת משפחה. ביטול זול (לדלג על האימון האחרון) **כבוי**.

**משפט אחד.** v2 הוכיח שהתלמיד יכול ללמוד. הוא הקפיא את שיעורי הבית הטובים הראשונים מוקדם מדי, ומעולם לא הראה לתלמיד את שאלות המבחן הקשות.

---

### v3 — אימון אחרון, עדיין רץ (מ־16 בספטמבר 09:55)

**מה שינינו (תיקונים לחורים של v2)**

- **מושל:** מקפיאים צילום כשהליכת ארגמקס דטרמיניסטית על שתי רשתות־בדיקה משתפרת. האימון **לא** נעצר. סבלנות וקיפול אחורה מסתכלים על הבדיקה הזו, לא על מקסימום בר־מזל של ארבע רשתות.
- **קצה הסף:** שני **מעברי** דחיסה באימון (ה־TEST מריץ שניים). השחקן רואה „הריווח נגמר, עצור.“
- **עלות קבוצה** במצב (חלק פרמטרים/MAC של כל הקבוצה המצומדת).
- **משך חיים:** לפחות 250 פרקים; סבלנות 150 על הבדיקה; קיפול אחורה לעילית עד 3 פעמים.
- **קטלוג:** 24 רשתות אימון (רוחבים בין סט האימון הישן לבין המוחזקות הצרות). בלי חפיפה עם קטלוג TEST.
- **ארבעה תפריטים אחים**, כל אחד 5 כפתורים `זהות | (0.9/0.8) × {L1, דירוג}`: דירוג = FPGM, SVD, או סולם־BN, כולם עם גמול מכויל; ועוד FPGM × **גמול NEON מקורי בלי כיול** (התא ההוגן „האם זה התפריט או הסקאלה“ ש־v2c לא נתן לנו על התפריט הזה).

**מה יודעים מטלמטריית אימון (לא TEST)** — נכון ל־08:55, ~23 שעות

- המבקרים חיים בזרועות הגמול המכויל. זרוע NEON הלא־מכוילת שרדה את כלל הקריסה שבו השתמשנו להרוג את v2c, ואז **התחילה לחתוך** בבדיקה השנייה — ובבדיקה הרביעית חזרה לזהות.
- הצילומים הקפואים הראשונים כבר מדללים על רשתות הבדיקה, ולכן הם ניתנים ל־TEST. אף אחת משלוש זרועות הדירוג המכוילות **לא עברה את הבדיקה הראשונה**. חלון שש הבדיקות (~פרק 70) עדיין פתוח. **לא** עוצרים אותן כי הלוח שנה אומר 17 בספטמבר.
- הקצב איטי מ־v2b (~2 פרקים/שעה מול ~5.6). אחרי לילה v3 בפרק ~45–52; v2b כבר היה אחרי 100. המושל עושה את עבודתו: האימון ממשיך אחרי ההקפאה הראשונה, בדיוק מה שהרג את v2b בפרק 15 מתוך 116.

**מה חסר עד שיגיע TEST.** האם מישהו מזה הולך על רשתות מוחזקות טוב יותר מ־L1/עדין **דו־מעבר** פעם־לקבוצה — הבקרה ההוגנת, כי v3 מתאמן בשני מעברים.

**משפט אחד.** v3 הוא v2 עם מורה שלא שולח את התלמיד הביתה אחרי החידון הטוב הראשון, ושסוף־סוף מראה את מחיר חיתוך זרם שיורי.

---

### V4 — שני ראשים: כמה, ואז אילו (מ־16 בספטמבר 12:04)

**מה חסר ב־v3.** תפריט משותף של 5 כפתורים עדיין קושר „כמה“ ל„איזה ממיין“ כצמד אחד. לגדול לחמישה כללי דירוג כ־13–15 כפתורים משותפים מדלל קרדיט (L1/L2/SVD הם כמעט אותו ממיין).

**מה V4 עושה.** מקודד משותף אחד; שתי רשימות. ראש 1: השאר 100 / 90 / 80%. ראש 2: L1, FPGM, סולם־BN, SVD, או טיילור. הדירוג **כבוי** בזהות (אין בחירה מעושה כשלא חותכים). טיילור הוא הממיין היחיד שמסתכל על אצווה של נתונים (`|w · גרדיאנט|`). L2 בחוץ (מתואם מדי עם L1).

**מצב.** רץ, פרק ~59, PPO-15. הבדיקה עלתה 0.115 → 0.210 בהקפאה השנייה, ואז נשארה 0.210 בבדיקות 3 ו־4. הצילום הניתן ל־TEST עדיין פרק 23. המבקר ירד לשלילי ב־PPO-13 והתאושש. עדיין אין TEST.

**משפט אחד.** V4 הוא הסוכן הראשון שיכול לומר „חתוך 20%“ ו„תשתמש ב־FPGM“ כשתי מחשבות, לא כצמד מוכן מראש.

---

## לילה (16–17 בספטמבר) — טלמטריה, לא TEST

v2b עזב אחיד בפרק 8, הקפיא צילום ניתן ל־TEST בפרק 15, ו**מת** ב־116 כי ציון אימון רועש לא עבר את המקסימום. v3/V4 נבנו כדי שהקפאה לא תעצור את העבודה. אחרי לילה הלולאה עובדת — והבדיקה שאמורה *להשתפר* עדיין לא.

ציון בדיקה = דחיסה ממוצעת (`1 − השארה`) של הליכת ארגמקס על שתי רשתות בדיקה צרות. גבוה יותר = דחוס יותר. צילום חדש קופא רק כשהציון הזה עולה.

| זרוע | בדיקה 1 (פרק 12) | 2 (24) | 3 (36) | 4 (48) | צילום ניתן ל־TEST |
|---|---|---|---|---|---|
| v3 FPGM | **0.262** | 0.000 זהות | 0.241 | (הולך) | פרק 11 |
| v3 SVD | **0.262** | 0.210 | 0.210 | (הולך) | פרק 11 |
| v3 סולם־BN | **0.210** | 0.210 | 0.202 | (הולך) | פרק 11 |
| v3 FPGM × NEON-גולמי | 0.000 זהות | **0.262** | 0.262 | 0.000 זהות | פרק 23 |
| V4 מפורק | 0.115 | **0.210** | 0.210 | 0.210 | פרק 23 |

**טוב.** המבקרים בזרועות המכוילות בריאים. האימון לא נעצר בהקפאה הראשונה (הכשל של v2b). תא NEON הגולמי לא מת כמו v2c. בקרת העדין הדו־מעבר עכשיו TEST: רשת קלה **−3.4 @ 0.536**, רשת קשה **−6.6 @ 0.923** (אותה השארה כמו עדין חד־מעבר).

**רע.** אף בדיקה מאוחרת לא עברה את ההקפאה הראשונה בשלוש זרועות הדירוג המכוילות. הארגמקס של NEON-גולמי חזר ל„אל תעשה כלום“ בבדיקה 4. V4 תקוע בהשארת רמת־עדין על רשת הבדיקה הצרה. הקצב ~פי 3 איטי מ־v2b, אז רצפת 250 הפרקים עדיין ימים משם. **אין עדיין TEST של v3/V4** — ה־QOS מלא באימונים.

---

## מנגנונים, במקום אחד

**דגימה מול ארגמקס.** TEST ישנים משכו פעולה אקראית מהמדיניות (ונשירה הייתה דלוקה). מ־Path 3 והלאה: **ארגמקס**, שחקן במצב הערכה. למדיניות עם שיא השניים לא מסכימים; למדיניות אחידה הארגמקס הוא פשוט ההטיה (עדין). טבלאות התזה משתמשות בארגמקס.

**איך בוחרים את המספר המצוטט.** הולכים על רשת ה־TEST. בכל צעד רושמים שינוי דיוק בולידציה וגודל. מצטטים את הנקודה הדחוסה ביותר שעדיין בתוך τ ב־**ולידציה**. מדווחים את שינוי הדיוק ב־**מבחן** שלה. לא משוטטים על סט המבחן. מדלגים על ממוצעי מעטפת, דיוק טוען־האימון, וקצוות שהולידציה שלהם כבר יצאה מ־τ.

**גמול.** שומרים על צורת NEON: קרדיט בתוך הסף, עונש מחוץ לסף, בונוס אופציונלי לרווח דיוק. v2a/v2b/v3־cbrt/V4 מנקד את חיתוך הפרמטרים/FLOP **שהתממש**, מכויל שורש־שלישי, כדי שחיתוך 20% על גזע בן 4 ערוצים לא יקבל אותו בונוס כמו חיתוך 20% על זרם שיורי רחב. v2c ו־v3־neonraw משתמשים בנוסחת NEON המקורית לפי **שיעור נומינלי**, בלי כיול. הכיול הוא מה שגרם למבקר ללמוד; הוא לא מוכיח שהטריכוטומיה שגויה. טרם קיבלתי תשובה לשאלות על פונקציית הגמול ששלחתי. אשמח עדיין לשיקול דעתך לפני שננעל את נוסח התזה.

**אילו מסננים.** הממיין ברירת המחדל הוא L1 (Li et al., ICLR 2017). v2b/v3 נותנים לסוכן לבחור L1 מול שותף אחד (FPGM / SVD / סולם־BN). V4 בוחר מבין חמישה, כולל טיילור (Molchanov et al.). היוריסטיקות משתמשות באותה סביבה, כך ש„מנצח חמדן“ הוא השוואת **לוח־זמנים** אלא אם התפריט עצמו בוחר דירוג.

**חיתוך פעם לקבוצה.** זרמי ResNet/DenseNet חולקים רוחב. לחתוך כל בעלים הוא צוק. חיתוך אחד לקבוצה למעבר הוא המנוף שהזיז את השארת r20 של Path 3 מ־~0.48 ל־0.746. אימוני v2+ ו־TEST עם זה דלוק.

**מושל.** v2 הקפיא על ציון אימון בר־מזל של ארבע רשתות ומת על סבלנות. v3/V4 מקפיאים עותק כשבדיקת ארגמקס על שתי רשתות משתפרת, וממשיכים לאמן (קיפול אחורה לעילית אם הבדיקה מתיישנת).

---

## איפה אנחנו אחר הצהריים (17 בספטמבר 14:50)

| רכיב | מצב |
|---|---|
| TEST של v2b | דק, שונה, דומה (בלי r32), C100 — בפנים. DenseNet עדין־דומה עדיין הולך (צעד ~91). DenseNet L1־דומה הוקרב ל־GPU. |
| בקרי יוריסטיקה דו־מעבר | שניהם דק **בפנים**. עדין §93: r20 **−3.4 @ 0.536**, r56 **−6.6 @ 0.923**. L1 §94: r20 **−7.3 @ 0.417**, r56 **−7.8 @ 0.898**. |
| TEST דק ראשון של v3 | **21428727 הסתיים §95.** fpgm פרק 11: r20 **−5.1 @ 0.536**, r56 **−6.8 @ 0.923**. אותה השארה מול עדין דו־מעבר; 1.7 / 0.2 נק׳ גרוע יותר. השארת עדין משובטת. svd TRAJ **21433272** ממתין. |
| v3 (4 זרועות) + V4 | באימון. V4 בדיקה 6 **0.241** (הראשונה שעברה את ההקפאה הראשונה). הקפאת הלוח שנה **סמלית**. |
| DRL על ImageNet | לא התחיל. C10/**C100 באימון** קפוא → ImageNet הוא מבחן הסט המאוחר. |
| ניצחון גלעד מול יוריסטיקות | **לא עמדנו** כטענת משפחה. ביטול זול כבוי. |

### המשך בעל־פה מהישיבה (~15:21) — לתיעוד

**1. „האם מאמנים מחדש את השכבה אחרי דילול?“** שלושה מתכונים, לא אחד. (א) מה שרץ היום: כוונון עדין לכל הרשת, המסננים שנשארו **נשמרים**. (ב) מה שעניתי בהתחלה: להקפיא את הרשת ולכוונן רק את השכבה המדוללת **עם המשקלים שנשמרו** — דגל קיים אצלנו; **0/32 תקינים** על ResNet-20/56. (ג) מה שהזכרת מ־NEON: להקפיא את הרשת, **לזרוק** את המסננים שנשארו בשכבה שנערכה, לאתחל מחדש, ולאמן **עד התכנסות**. הקוד של SPECTRA כרגע ממפה את הנתיב הזה לדילול־ושמירה רגיל. ניישם את (ג), נאמן סוכן כך, ונבחן מול הטוב הנוכחי. המימוש בישיבת Fable; לא נשכתב את אימוני v3/V4 החיים.

**2. ההערכה חייבת לפי מוסכמת דילול CNN.** אחרי הוכחת ערך, הוועדה תבקש את SPECTRA ליד SOTA על **הרשתות שלהם**. נריץ את הסוכן הקפוא **ואת** היוריסטיקות באותה לולאה על תאי־בית (לפחות ResNet-56 CIFAR-10 במקור ~94% — לא w4 הרזה), ונדפיס `מקור | אחרי דילול | Δacc | פרמטרים שנשמרו | FLOPs שנשמרו`. נצטט DepGraph / OCS / FPGM / HRank / Slimming / ATO עם כיתוב מקור וכוונון. לא נטען לניצחון על SOTA ממוקד בתא הבית שלו.

סיוטות האלגוריתם לכל שלב (Path 3 → V5) ב־`docs/paper/LOOP_ALGORITHMS.md` לסימון.

---

## 30 בספטמבר — האם קטלוג TEST מלא ישים?

**אם „קטלוג מלא“ פירושו כל זרוע v3/V4 × דק + דומה + שונה + CIFAR-100:** **לא.** קטלוג דומה אחד הוא כיומיים על GPU אחד. חמישה שחקנים × ארבעה קטלוגים זה יותר זמן־GPU מ־13 ימים כשחמישה GPUs עדיין באימון. לאימונים עצמם יש רצפת 250 פרקים (עוד כמה ימים בקצב הנוכחי).

**אם „קטלוג מלא“ פירושו: לסיים את בקרי v2b, להריץ יוריסטיקות דו־מעבר, ולבחון את צילומי v3/V4 הקפואים על שתי הרשתות הדקות (ואולי המשפחה השונה):** **כן, אם ה־GPUs ממשיכים לזוז.** שתי יוריסטיקות הדו־מעבר בפנים. TEST דק ראשון של v3 בתור. משפחה דומה ל־v3/V4 עד ה־30 היא מתיחה, לא תוכנית. האימונים ב־~50 מתוך 250 פרקים — עוד כמה ימים.

### למה אפשר עדיין לכוון בהגשה ב־30 בספטמבר (אם TEST דק של v3/V4 נוחת)

תזת **שיטה** שאפשר להגן עליה, לא תזת העברה גמורה:

- הפרוטוקול ישר (ארגמקס, בחירה בולידציה, פעם־לקבוצה, רק TEST מצוטט).
- Path 3: הסוכן הקפוא הישן ≡ עדין / צוק־חמדן; דגימה ניפחה את הסיפור.
- v2b: מדיניות גנרית עם שיא, לראשונה; בתוך הסף על משפחות CIFAR-10 דומות; שונה ≡ השארת L1; C100 זהות.
- v3/V4: אימון אחרון מוגדר; TEST דק אם יושלם; ראשוני אם לא ינצח יוריסטיקות דו־מעבר.
- מטריצת כיסוי: C10 דומה/שונה מ־v2b; C100 של סוכן C10־בלבד הוא **מדידה**, לא טענת ההעברה. הסוכן הבא מתאמן על משפחות × סטים מגוונים. פארטו: v2b מול עדין מול L1 מול Path 3, ועוד כוכבי ספרות **מצוטטים**, לא רצים מחדש.
- טענה שאפשר לעמוד מאחוריה: *בנינו לולאת דילול DRL שעוברת בין רשתות ושבאמת לומדת, ומדדנו אותה בהגינות.* טענה שאי אפשר לעמוד מאחוריה: *הסוכן האחרון מנצח יוריסטיקות על משפחות מוחזקות, והדירוג הוא הסיבה.*

### מה מוותרים אם עומדים ב־30 בספטמבר

- v3/V4 כסוכן התזה (אימון לא גמור; רוב מבחני ההחזקה חסרים).
- טענת דירוג־כפעולה (v2b כבר נראה כמו L1 על השונה; מבחני v3/V4 הם המבחן למשפט הזה).
- בקרי דו־מעבר על קטלוגי דומה/שונה.
- זמן לכתוב, לקבל הערות, ולשים את מטריצת הכיסוי ואת הפארטו בטיוטה באיכות תזה.
- כל מבחן העברה ל־ImageNet.
- מבט שני אם בדיקת v3 נתקעת — קיפול אחורה וצילומים מאוחרים לא ייכנסו לטבלאות.

---

## בשביל מה החודש הנוסף (עד 30 באוקטובר)

לא „עוד רעיונות.“ לסיים את הלולאה שכבר הוגדרה:

1. לתת ל־v3/V4 להגיע למשך החיים המתוכנן (או קיפול/סבלנות), לבחון צילומים קפואים על **דק**, אחר כך **דומה**, אחר כך **שונה**, CIFAR-100 בסוף. בקרות הוגנות = עדין ו־L1 דו־מעבר פעם־לקבוצה.
2. לסגור את G1: DenseNet דומה עדין (ואם רוצים את השלישייה, גם L1), כדי ש־v2b מול יוריסטיקות יהיה טבלת משפחה שלמה.
3. אם לשתי זרועות אימון אחרון או יותר יש TEST דק: **תיק ולידציה** (לכל רשת, השחקן עם נקודת הולידציה הטובה; מדווחים את דיוק המבחן שלו). כיתוב: לא מיזוג משקלים.
4. לכתוב את התזה במסגרת 18 באוגוסט: כיסוי + פארטו, העברה, בלי התרברבות SOTA־בית. לנעול פרוזת גמול אחרי הישיבה הזו.
5. אופציונלי אם מתפנה GPU: סוכן C10 קפוא על ImageNet כמבחן העברה קצר — עדיין **בלי** אימון DRL על ImageNet.

זה חודש של TEST + כתיבה, לא תוכנית מחקר חדשה.

**אמצע דצמבר** הוא שעון האוניברסיטה: זמן לקבל הערות, ללטש את מטריצת הכיסוי ואת הפארטו לאיכות תזה, ולספוג החלקה אם מבחני v3/V4 נמשכים. שני חודשים וחצי מספיקים לסיים את הסבב שכבר הוגדר ולכתוב תזה שלמה. זו לא תוכנית מחקר שנייה.

---

## למה אני מבקש (נימוקים, קצר)

1. **האובייקט המדעי השתנה בשבועיים האחרונים.** לפני Path 3 היינו מגישים מדיניות אחידה כסוכן DRL גנרי. זו הייתה תזה שגויה.
2. **האופטימייזר עובד; טבלאות ההעברה לסוכן *הנכון* עדיין לא בפנים.** הגשה עכשיו מתעדת אבחון ועוד v2b, ומשאירה את האימון האחרון בלי TEST.
3. **תנאי הניצחון של גלעד עצמו הוא השארה שווה מול יוריסטיקות.** לא עמדנו בו. האימון האחרון הוא הניסיון. להרוג אותו בגלל הקפאת לוח שנה מבטיח שלעולם לא נמדוד אותו.
4. **האימון האחרון כבר מוגדר ורץ.** החודש הנוסף הוא הערכה וכתיבה, לא ציד שיטה חדש (אין גמול חדש, אין DRL על ImageNet, אין אתחול מקודד מחדש).
5. **שני שעונים, תוכנית אחת.** 30 באוקטובר מספיק ל־TEST ולטיוטה שלמה ראשונה. אמצע דצמבר היא הדחייה האוניברסיטאית (~חודשיים וחצי) כדי שהערות וליטוש לא יידחקו. הטבלאות היום עדיין אינן התזה שהיית רוצה לחתום עליה; גם ב־30 בספטמבר לא יהיו.

---

## הבקשה

אבקש שתתמוך ב**שני** השעונים:

1. להחזיק את **30 באוקטובר 2026** כתאריך התזה **לעבודה** (כרטיס ניסוי + טיוטה שלמה ראשונה).
2. לתמוך בבקשה לאוניברסיטה לדחות את מועד ההגשה בכ**שני חודשים וחצי**, ל־**אמצע דצמבר 2026**.

הסיבה אינה דחייה לשמה. יש עכשיו מדיניות עם שיא, אימון אחרון שמוגדר וכבר רץ, ופרוטוקול מבחן שאינו בוחר על סט המבחן. 30 באוקטובר הוא הניסוי. אמצע דצמבר הוא כתיבה והליך.

אני זמין לדון בפונקציית הגמול בכל עת. האימון האחרון מתקדם עם הגמול המכויל כברירת מחדל ועם גמול NEON המקורי כבקרה מפורשת, אלא אם תעדיף אחרת.

אשמור על מסגרת 18 באוגוסט, אצטט רק TEST, ולא אטען לניצחון משפחתי על יוריסטיקות עד שהטבלאות יראו זאת. v3/V4 ממשיכים לרוץ עד שנבחן אותם או שתאמר לי לעצור.

בכבוד רב,  
עידו פרצקי
