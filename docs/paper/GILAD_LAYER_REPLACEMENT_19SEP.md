# Note for Dr. Gilad Katz — NEON layer replacement on CNNs

**From:** Ido Paretsky  
**Date:** 19 September 2026  
**What this is:** a short status of the idea you described (throw away the remaining weights of the layer we just shrank, train that new layer until it converges, freeze the rest). We ran it on CNNs **without a learned agent**, so the only thing that changes between arms is the recovery method. Please fill nothing; the last section lists three questions we need from you. Fable will add its 19 Sep read in the marked slots before this reaches you if those slots are still blank.

Numbers are **test-set** accuracy drop at the most compressed point that still stays inside the allowed validation drop (τ = 10 percentage points). Size is **fraction of parameters kept**. All rows are preliminary.

---

## 1. What we compared

Three recovery methods, **same pruning walk** (always keep 90% of each legal channel-group, two passes, one edit per group per pass):

| Recovery method | Weights after the cut | Who is trained | Until when |
|---|---|---|---|
| **SPECTRA today** (keep leftover filters) | surviving pretrained filters stay | whole network | 40 epochs, patience 10 |
| **Your NEON idea, CNN group** | surviving filters thrown away; producers, the next layer’s input slice, and group-norm are drawn fresh | only that group; rest frozen | validation accuracy plateaus (patience 6, cap 60) |
| **Your NEON idea + a short whole-net polish** | same throw-away | group to plateau, then the whole net at 0.1× learning rate | polish: 8 epochs, patience 3 |

On dense / fully-connected NEON this throw-away **is** the paper method (Hirsch & Katz 2022, §3: “generate a new layer of the desired dimensions… weights initialized randomly… freeze all layers except the new one… train until convergence”). SPECTRA had been doing the method you **rejected** there: keep leftover weights.

We measured on:

- skinny CIFAR-10 ResNet-20-width2 and ResNet-56-width4 (the cheap diagnostic pair);
- the committee ResNet-56 CIFAR-10 (chenyaofo, origin 94.37%).

---

## 2. How the idea fared on CNNs

**Keep leftover filters** (SPECTRA today) on the skinny pair, two-pass 90% walk: ResNet-20 **−3.4 pp at 53.6% kept**; ResNet-56 **−6.6 pp at 92.3% kept**.

**Throw-away, group only** (your idea, source-literal: also re-draw the consumer):

| Net | Test Δacc @ params kept | Read |
|---|---|---|
| skinny ResNet-20 | **−0.9 @ 98.8%** | almost no cut — the selected point is near the original net |
| skinny ResNet-56 | **−0.1 @ 99.9%** | same |
| chenyaofo ResNet-56 | **−0.5 @ 99.9%** | same |

**Throw-away + whole-net polish:**

| Net | Test Δacc @ params kept | vs keep-leftover |
|---|---|---|
| skinny ResNet-20 | **−10.3 @ 88.4%** | shallower cut **and** 6.9 pp worse |
| skinny ResNet-56 | **−0.2 @ 99.9%** | still almost no cut |
| chenyaofo ResNet-56 | **−1.0 @ 99.9% kept**, val −0.23 pp | empty band — polish did not open a cut |

**Throw-away, producers only** (oral reading: re-draw the pruned layer’s filters + norms; leave the next layer’s incoming weights; no polish). Skinny pair only; job **21459742**, 20 Sep:

| Net | Test Δacc @ params kept | vs full-group throw-away |
|---|---|---|
| skinny ResNet-20 | **−0.7 @ 98.8%** | same empty band |
| skinny ResNet-56 | **+0.0 @ 99.9%** | same empty band |

**How hard we can press this (20 Sep).** Combined matrix, all **no-agent**, same 90% two-pass walk, quote test Δacc @ params kept:

| Net | Keep leftover (SPECTRA today) | Throw-away, group | Throw-away + polish | Throw-away, producers only |
|---|---|---|---|---|
| skinny ResNet-20 | **−3.4 @ 53.6%** | −0.9 @ 98.8% | **−10.3 @ 88.4%** | −0.7 @ 98.8% |
| skinny ResNet-56 | **−6.6 @ 92.3%** | −0.1 @ 99.9% | −0.2 @ 99.9% | +0.0 @ 99.9% |
| chenyaofo ResNet-56 | **−3.9 @ 66.1%** | −0.5 @ 99.9% | −1.0 @ 99.9% | *(not run; skinny pair already ≡ group)* |

Show this table. Caption it **preliminary, no learned agent**. It is enough to say: on this walk, the dense-NEON throw-away recipe does not recover a real CNN cut, and keeping leftover filters does. It is **too soon** to write as a validated thesis fact that “the DNN approach cannot transfer to CNNs”: we have not TESTed a learned agent under throw-away, we have not rematched 200-epoch SGD, and “until convergence” on train loss vs validation is still Q3.

**Failure points on CNNs (vs NEON’s success on dense nets).**

On a dense stack, the new layer is a self-contained Linear: freeze everything else, train it, and the rest of the net is unchanged. On a residual CNN that is not true:

1. **Skip connections.** The block output is `F(x) + x`. A freshly random `F` is added to a frozen identity path. Early in group training the residual is noise; BatchNorm on the frozen rest sees the wrong activation scale.
2. **The “consumer” is half the next block.** NEON’s source also replaced the *next* Linear (`nn.Linear(new_size, out)`), not only the pruned layer. On a ResNet, every block’s first convolution reads the previous block’s channels, so replacing “the group” rebuilds about **half a stage from scratch** inside a frozen network.
3. **BatchNorm.** Frozen BN running stats were collected with the old filters. After throw-away they are wrong; we hold frozen BNs in eval mode, but the mismatch remains.
4. **The walk cannot “choose not to cut.”** The 90% policy always cuts. If throw-away cannot recover that cut, validation falls off a cliff and the selected point of the trajectory stays at **identity** (kept ≥ 98%). That is what we see on all three nets without polish, and on skinny ResNet-56 even *with* polish.

So the dense-DNN success does not automatically transfer. SPECTRA’s current keep-leftover recipe **does** recover a real in-band cut on the committee ResNet-56 (**−3.9 pp at 66.1% kept**). The throw-away recipe has not, yet.

---

## 3. What Fable expected, what already happened, surprises

**Expected, and it happened**

- If throw-away cannot recover a CNN group, the 90% walk’s best in-band point stays near the original size. That is exactly the skinny pair and the committee ResNet-56 without polish.
- Under the live training reward, a 20-point *legal* cut is worth about **2.7** after cube-root, and a 20-point miss is **−20**. About eighteen legal cuts pay for one miss. A policy that always keeps 90% is then the safe optimum. The factored-head actor we TESTed (snapshot episode 83) **copied that 90% keep** on both skinny nets (ResNet-20 **−3.7 @ 53.6%**, ResNet-56 **−6.9 @ 92.3%** — 0.3 pp worse than the 90% heuristic at the same size). Fable called this before the TEST.
- Accuracy *increases* after prune+fine-tune almost never happen on these 90–96% CIFAR-10 nets (0 of ~17 700 non-identity training steps). A bonus for “accuracy went up” would not have fired.

**Expected, and it has not happened (yet)**

- Fable’s assigned CNN method was throw-away **plus** the short whole-net polish, on the guess that skip-adds and frozen BN need a coupling step. On **all three nets** the polish did **not** match keep-leftover: skinny ResNet-20 got a cut but was much worse; skinny ResNet-56 and the committee ResNet-56 stayed at identity.

**Surprises**

- **Bad.** Skinny ResNet-20 with throw-away + polish was **−10.3 pp at 88% kept**, against **−3.4 pp at 54% kept** when we keep leftover filters. Not a close miss: worse accuracy at a larger model.
- **Good / mixed.** Keep-leftover on the 94.37% ResNet-56 **did** find a real 34% parameter cut inside the band. The walk is not broken on that net; only throw-away is.
- **Good.** One ranking-menu train (BN-scale) later beat its own first freeze (new snapshot at episode 107, same probe score 0.262). The rewind-to-best rule is doing something. We have not TESTed that later snapshot yet.
- **Infra, not science.** The CIFAR-100 “which nets are recoverable enough to enter the next train set” job died in 31 seconds on a missing file. No scientific verdict. We are fixing that separately.

---

## 4. Three questions for you

### Q1 — Training reward, in-band arm

**The issue.** NEON’s three-way reward cubes the size of a cut when accuracy *rises* or when the drop exceeds τ, and leaves a legal in-band drop **linear**. We then take a cube-root of the whole thing so the critic can regress it. That cube-root also shrinks the already-linear in-band arm: a 20-point legal cut becomes ~+2.7, a 20-point miss stays −20. The 90% heuristic is then what a rational agent copies — and that is what our TESTed actors do.

**Ido (19 Sep):** GO. Overhaul toward a more explorative, more rewarding in-band signal. Isolated training job, default off, fresh actor. Do not change the six trains that are already running.

**Grok 4.6 (19 Sep):** GO, same cell. Keep cube-root on the *cubed* arms only; leave in-band linear. Then a 20-point legal cut is +20 and a miss is −20 (one-to-one). Same catalog, same optimiser, same ranking menu as the live FPGM train (`21385158`) — only this map changes. Do not also add entropy bonuses or a “accuracy went up ×2” term in the same job (the rise arm is empty on these nets).

**Fable (18 Sep sitting, before the new TESTs):** this was the strongest untested lever; ~15 lines; GO/no-go asked of Ido. Ido has now said GO.

**Fable (21 Sep, after the new TESTs):** the change was made and trained (job 21459737: the running FPGM recipe with only the reward map changed — in-band arm linear, cube-root kept on the two cubed arms). Its first frozen policy has been TESTed (ledger §111). Two results:

- On the skinny ResNet-56 the policy **cut past the 90 % point for the first time**: **−7.1 pp at 75.6 % kept**, inside the validation band. The 90 %-heuristic stops at 92.3 % kept (−6.6 pp); the "always cut hardest" heuristic leaves the band beyond 89.8 % kept (−7.8 pp). Every earlier learned policy, whatever its ranking menu or head, stopped at exactly the heuristic's 92.3 %. The only other policy that had cut deeper on this net was the one trained with your original **uncubed-root** reward (also a linear in-band arm, −7.4 pp at 75.7 %). So the arithmetic in "the issue" above was the right diagnosis on the net that can move: the cube-root on the in-band arm was what kept every agent at 90 %.
- On the skinny ResNet-20 nothing changed (−3.5 pp at 53.6 %), and I now think **nothing can**: its layers are 2, 4 and 8 channels wide, so for most groups only one cut size is legal and every non-trivial policy produces the identical walk. That net tells us whether a policy cuts at all, not how well it chooses. We should stop reading it as a comparison of policies.

What it is not yet: a 2 pp win at equal size. No heuristic in our loop reaches 75 % kept on that net inside the band, so the fair comparison (three passes of the 90 % heuristic, and of the hardest-cut heuristic) is the next cheap control, not a new method. The policy itself is also not very decisive — its probabilities sit close to uniform over the legal rates — yet its deterministic choice produces the deeper walk. My recommendation: make the linear in-band arm the default for **new** trainings (the running ones stay as controls), keep the cube-root on the cubed arms only, and judge the agent on the matched-size heuristic rows. The training that keeps the fine-tune at the TEST budget (40 epochs instead of 12) has meanwhile produced the first snapshot whose selection score rose above the ceiling all 12-epoch trainings share; its TEST is running now.

### Q2 — What “the layer” means on a CNN

**The issue.** Your oral wording was: reinitialise the remaining filters of the **newly pruned layer**. The NEON **source** also rebuilt the next Linear (the consumer). On a residual stream that consumer is the next block’s first convolution, so the source-literal CNN reading redraws about half a stage. We ran the source-literal reading. We have **not** yet run the oral reading (re-draw only the pruned layer’s filters and its norms; leave the next layer’s surviving incoming weights to adapt).

**Ido (19 Sep):** enqueue that oral-reading ablation after the current group-scope walks finish. The last of those (committee ResNet-56 + polish) has now completed empty-band; the ablation is the next no-agent walk.

**Grok 4.6 (19 Sep):** same. The empty-band result is compatible with “throw-away does not recover CNNs” **and** with “we threw away too much by rebuilding consumers.” Those two are not distinguishable until the oral-reading walk exists. Default for now remains the source-literal group. Recommendation: run the cheap skinny ablation, then you pick the quote.

**Grok 4.6 (20 Sep):** the oral-reading walk **21459742** has now completed. Skinny ResNet-20 **−0.7 @ 98.8%**, skinny ResNet-56 **+0.0 @ 99.9%** — the same empty band as full-group throw-away. The two stories **are** distinguishable: rebuilding consumers was not the failure mode. Throwing away the surviving filters of the pruned layer is enough. Quote the source-literal group as the paper method; caption the oral reading as an ablation that also failed. Still **not** a learned-agent result. Do not start a throw-away DRL train from this row.

**Fable (18 Sep sitting):** default = source-literal group; ask Gilad; one extra skinny walk if wanted.

**Fable (21 Sep, after the new TESTs):** the oral reading has now been run and it lands in the same place as the source reading (skinny ResNet-20 −0.7 pp at 98.8 %, skinny ResNet-56 +0.0 pp at 99.9 %). That settles the question the two readings could have separated: the failure is **not** that we rebuilt too much. Throwing away the *surviving* filters of the pruned layer is by itself enough to lose the cut. I would quote the source reading as NEON's method in the paper and mention the oral reading in one sentence as an ablation that failed the same way.

Why I think the dense success does not carry over, beyond the skip/BatchNorm points in §2: in a dense network the shrunk layer is a linear map on a fixed feature vector, so a freshly initialised layer can re-learn what the discarded neurons did from the frozen features around it in a few epochs. In a CNN the surviving filters we throw away are, by construction of the ranking, the filters that carried most of the layer's function, and the frozen layers downstream were co-adapted to exactly those spatial features. Retraining one group from random inside a frozen hierarchy cannot recover that in the budget we gave it — and the budget was not the problem: the group training stopped improving after a median of 26 epochs out of 60, i.e. it converged to a worse solution rather than running out of time. Keeping the surviving filters and fine-tuning the whole network is the recipe that works on CNNs, and it is the one we have been using.

The one thing these walks cannot prove is what a *learned* agent would do under throw-away, because the 90 % walk cuts every group it is allowed to. An agent could learn to cut only groups that throw-away can recover. But a training reward under throw-away would pay for "do not cut" on almost every step, so I expect such an agent to learn not to prune. I do not recommend spending a GPU on that training; the table above is the identifying evidence, captioned as preliminary and without an agent.

### Q3 — “Until convergence”: training loss or validation accuracy?

**The issue.** The paper says train the new layer until convergence. The NEON **source** stops on **training-loss** patience 10 (no validation inside that fine-tune) and restores the best-loss weights. SPECTRA’s reward is defined on **validation** drop vs the original net, so we defaulted the CNN port to a **validation** plateau (patience 6, cap 60). A flag replays the source (train-loss) if you want that quoted.

**Ido (19 Sep):** no decision yet — this question is for you.

**Grok 4.6 (19 Sep):** keep **validation** as SPECTRA’s default, because that is the quantity the agent is scored on. Offer train-loss as a labelled ablation, not as a silent swap. Ask you which rule the paper should name as NEON’s.

**Fable (18 Sep sitting):** defaulted to validation; asked whether it is OK to quote validation.

**Fable (21 Sep, after the new TESTs):** my recommendation is to write both in one sentence — "NEON's code stops when the training loss has not improved for 10 epochs; our CNN port stops when the validation accuracy has not improved for 6 epochs, because validation accuracy is what the reward is computed from" — and to quote the validation rule as SPECTRA's. On a dense network the two rules agree in practice: a small new layer's training loss and validation accuracy move together. They only come apart when the new layer starts to overfit the training split, and on that case the validation rule is the *more* charitable one for the throw-away idea (the train-loss rule would keep training into overfitting and make it look worse). So switching to the train-loss rule cannot rescue the results in §2; a flag exists to replay it if you want it quoted, but I would not spend a GPU on it.

---

## 5. What we are not asking you to decide here

Live training jobs keep running. We are not stopping them. We are not claiming to beat focused CNN-pruning methods on their home cell. The next learned agent that attacks the “always keep 90%” clone is the in-band-linear reward job (Q1), not another ranking menu.

---

# עברית

**אל:** ד"ר גלעד כץ  
**מ:** עידו פרצקי  
**תאריך:** 19 בספטמבר 2026

זו הערה קצרה על הרעיון שתיארת: אחרי שגוזרים שכבה, **זורקים** את המשקלים שנשארו, מאתחלים שכבה חדשה ברוחב החדש, מקפיאים את שאר הרשת, ומתאמנים על השכבה החדשה עד התכנסות. זה מה שעבד ב-NEON על רשתות צפופות (לא קונבולוציה). ב-SPECTRA עד היום עשינו את ההפך: **שמרנו** את הפילטרים ששרדו וכיילנו את כל הרשת.

בדקנו את זה על רשתות קונבולוציה **בלי סוכן לומד** — אותה הליכת גזימה (תמיד שומרים 90% מכל קבוצת ערוצים, שני מעברים). שלוש שיטות שחזור:

1. **SPECTRA היום** — שומרים פילטרים, מכיילים את כל הרשת.  
2. **הרעיון שלך, כפשוטו בקוד של NEON** — זורקים, מציירים מחדש גם את השכבה הבאה (הצרכן), מתאמנים רק על הקבוצה עד שהדיוק בולידציה מתייצב.  
3. **אותו דבר + כיול קצר של כל הרשת** בקירוב למידה נמוך (פי 0.1).

## מה יצא

על ResNet-20 ו-ResNet-56 הרזים (CIFAR-10), שיטת "שמור פילטרים" נתנה **−3.4 נקודות ב-54% מהפרמטרים** ו-**−6.6 ב-92%**.  
שיטת "זרוק והתאמן על הקבוצה" כמעט **לא גזרה**: הנקודה שנבחרה נשארת סביב הרשת המקורית (99% מהפרמטרים). אותו דבר על ResNet-56 של הקומיטי (מקור 94.37%).  
עם הכיול הקצר: ResNet-20 הרזה יצא **גרוע יותר** (−10.3 נקודות ב-88% שמורים, מול −3.4 ב-54% כששומרים פילטרים). ResNet-56 הרזה ו-ResNet-56 של הקומיטי נשארו בלי גזירה אמיתית (99.9% מהפרמטרים).  
קריאת בעל-פה (מאתחלים רק את הפילטרים של השכבה שנגזרה, בלי הצרכן) רצה ב-20 בספטמבר על הזוג הרזה: **אותו פס ריק** (−0.7 ב-98.8% / +0.0 ב-99.9%). כלומר הזריקה נכשלת גם בלי לבנות מחדש את השכבה הבאה.

## למה זה נכשל בקונבולוציה אחרי שהצליח בצפופות

ברשת צפופה השכבה החדשה היא Linear סגורה. ברשת שיורית החיבור הוא `F(x)+x`: `F` אקראי מתווסף למסלול זהות קפוא, ו-BatchNorm הקפוא זוכר סטטיסטיקה של הפילטרים הישנים. בנוסף, בקוד של NEON הוחלפה גם השכבה **הבאה**; ב-ResNet זה אומר שמציירים מחדש בערך חצי שלב מתוך רשת קפואה.

## מה שפייבל ציפה, ומה שהפתיע

צפוי והתממש: אם הזריקה לא משחזרת קבוצת CNN, נקודת הולידציה נשארת ליד הרשת המקורית; וגם: פונקציית הגמול הנוכחית עושה את מדיניות "שמור 90%" לאופטימום בטוח — הסוכן שבדקנו העתיק בדיוק את זה.  
צפוי ולא התממש: הכיול הקצר של כל הרשת יישר קו עם "שמור פילטרים". הוא לא יישר על אף אחת משלוש הרשתות.  
הפתעה רעה: ResNet-20 עם זריקה+כיול היה הרבה יותר גרוע, לא "כמעט".  
הפתעה טובה: על ResNet-56 94.37%, "שמור פילטרים" **כן** מצא גזירה אמיתית (−3.9 נקודות ב-66% מהפרמטרים). ההליכה עצמה לא שבורה; הזריקה כן.

## שלוש שאלות אליך

**ש1. גמול בתוך התקציב.** אחרי שורש-שלישי, גזירה חוקית של 20 נקודות שווה בערך 2.7, וחריגה של 20 שווה −20. עידו אמר **קדימה**: להשאיר את שורש-השלישי רק על הזרועות שכבר בריבוע/בחזקה, ואת הזרוע שבתוך התקציב **ליניארית** (+20 מול −20). אימון חדש, בלי לגעת בשישה האימונים שרצים. פייבל ישלים כאן אחרי ה-TESTs החדשים אם השדה ריק.

**ש2. מהי "השכבה" ב-CNN.** בנוסח שבעל-פה: מאתחלים מחדש את הפילטרים של השכבה שנגזרה. בקוד: גם את הצרכן. הרצנו את קריאת הקוד. ב-20 בספטמבר רצה גם קריאת בעל-הפה על הזוג הרזה — **אותו פס ריק**. איזו קריאה לצטט בנייר? (המלצת גרוק: לצטט את קריאת הקוד כשיטת NEON, ולציין את בעל-הפה כבדיקה שנכשלה גם היא.)

**ש3. "עד התכנסות".** בנייר: עד התכנסות. בקוד של NEON: סבלנות על **הפסד האימון**, בלי ולידציה. אנחנו ברירת-מחדל על **ולידציה**, כי כך נמדד הגמול. מה לצטט כ-NEON?

האימונים החיים ממשיכים. אין כאן טענה שאנחנו מנצחים שיטות CNN ייעודיות על רשת הבית שלהן.
