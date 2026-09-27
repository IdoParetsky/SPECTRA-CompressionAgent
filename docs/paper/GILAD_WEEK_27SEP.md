# Note for Dr. Gilad Katz — week of 27 Sep 2026

**From:** Ido Paretsky
**Status:** draft for Ido’s review. Not sent.
**Since:** the 17 Sep meeting, your 19 Sep comment on the network list, and the two notes already written (layer replacement, 19 Sep; benchmarking setup, 21 Sep).
**Two subjects you asked about:** (1) throw away the pruned layer and train a new one, instead of fine-tuning the surviving weights; (2) a real experimental setup, not a grocery list of networks.

Numbers below are test-set accuracy change at the most compressed point that is still inside a 10-point validation budget. Size is the fraction of parameters kept. Preliminary.

---

## English — executive

**Throw-away.** On the original NEON loop this is not a training-only stage. Every pruning step, while the agent learns and again when the frozen agent is applied, replaces the layer with a new smaller one, draws its weights at random, freezes the rest, and trains that new layer. SPECTRA’s CNN version does the same draw (Kaiming normal, the usual new-convolution initialization, not zeros). On a forced 90% channel cut with no learned agent, throw-away does not recover a real CNN compression. Keeping the surviving filters does. Redrawing only the pruned layer, which matches the oral wording, fails the same way as redrawing the next layer too. A short polish of the whole network does not rescue it. We are not training an agent under throw-away. The next honest version of “generate a new layer,” if we try one, builds that layer from the old activations (a projection), or refits the next layer by least squares. It is not another random draw.

**Benchmark.** The setup in the 21 Sep note is now the locked protocol: three cells only (DepGraph’s CIFAR-10 ResNet-56, OCS’s CIFAR-10 VGG-16, DepGraph’s CIFAR-100 VGG-19); the agent is not trained on those architectures; two operating points (our 10-point rule, and their published size); budgets reported as GPU time. What moved this week is the yardstick, not a claim that we beat those papers. On the full-width ResNet-56 twin, keep-leftover finds a real cut (−3.3 points at 66% kept). On VGG-19 CIFAR-100 the same 2-pass walk’s selected point is the original network. No single fine-tune learning rate both admits CIFAR-100 and preserves the CIFAR-10 control, so CIFAR-100 is not in the training set yet. The best learned policy on the cheap ResNet-56 diagnostic keeps 76% of the parameters at −7.1 points, twice. Every earlier learned policy stopped at 92%, next to the simple 90% rule. That diagnostic is not the committee table.

---

## עברית — תקציר

**זריקת המשקולות.** ב-NEON המקורי זה לא שלב של האימון בלבד. בכל צעד גיזום, גם בזמן שהסוכן לומד וגם כשמפעילים את הסוכן הקפוא, מחליפים את השכבה בשכבה חדשה וצרה יותר, מגרילים את המשקולות, מקפיאים את השאר, ומאמנים את השכבה החדשה. ב-SPECTRA ההגרלה היא האתחול הרגיל של שכבת קונבולוציה חדשה (Kaiming), לא אפסים. על חיתוך כפוי של 90% בלי סוכן לומד, הזריקה לא משחזרת דחיסה אמיתית של CNN. השארת המסננים ששרדו כן משחזרת. ציור מחדש של השכבה הגזומה בלבד, לפי הניסוח בעל-פה, נכשל כמו ציור מחדש של השכבה הבאה. ליטוש קצר של כל הרשת לא מציל את זה. אנחנו לא מאמנים סוכן תחת זריקה. הגרסה הישרה הבאה של «לייצר שכבה חדשה», אם ננסה, בונה את השכבה מהאקטיבציות הישנות, או מתאימה את השכבה הבאה בריבועים פחותים. לא עוד הגרלה.

**מדידה.** המערך מהפתק של 21 בספטמבר נעול: שלושה תאים בלבד (ResNet-56 של DepGraph על CIFAR-10, VGG-16 של OCS על CIFAR-10, VGG-19 של DepGraph על CIFAR-100). הסוכן לא מאומן על הארכיטקטורות האלה. שתי נקודות הפעלה: הכלל שלנו (ירידה של עד 10 נקודות בולידציה), והגודל שהם פרסמו. התקציב הוא זמן GPU. השבוע זז סרגל המדידה, לא טענה שניצחנו את המאמרים. על תאום ה-ResNet-56 ברוחב מלא, השארת המסננים מוצאת חיתוך אמיתי (3.3− נקודות ב-66% פרמטרים). על VGG-19 ב-CIFAR-100 הנקודה שנבחרת באותו מסלול היא הרשת המקורית. אין קצב למידה אחד שמכניס CIFAR-100 וגם שומר על ביקורת CIFAR-10, ולכן CIFAR-100 עדיין לא בסט האימון. המדיניות שנלמדה הכי טוב על רשת האבחון הזולה שומרת 76% מהפרמטרים בירידה של 7.1 נקודות, פעמיים. כל מדיניות קודמת נעצרה ב-92%, ליד כלל ה-90% הפשוט. זו אבחנה, לא טבלת הוועדה.

---

## English — detail

### 1. Throw-away: what NEON did, and on which stages

Hirsch & Katz 2022, §3, inside the pruning loop (Algorithm 1), not as a separate post-training stage:

- **Layer replacement.** Rather than removing neurons, generate a new layer of the desired size. It replaces the analyzed layer. Its weights are initialized randomly.
- **Layer fine-tuning.** Freeze every layer except that new one. Train until convergence.
- The same loop is the offline training of the agent **and** the test phase, when the trained agent is applied to a network it did not train on. `action = 1` (no cut) skips both the replacement and the fine-tune.

The public source also rebuilds the *next* linear layer and a fresh batch-norm, and trains those three modules. Patience in the source is 10 epochs on the **training** loss, not on validation. The paper’s sentence names only the new layer.

SPECTRA’s live path is the method that paper rejected: keep surviving filters, fine-tune the whole network (recipe A). The CNN throw-away experiments change only the recovery, on a fixed 90% walk, with no agent.

| Recovery | Skinny ResNet-20 | Skinny ResNet-56 | Full ResNet-56 (chenyaofo) |
|---|---|---|---|
| Keep surviving filters | −3.4 @ 53.6% | −6.6 @ 92.3% | −3.9 @ 66.1% (twin rerun −3.3 @ 66.1%) |
| Throw-away, group, val patience | −0.9 @ 98.8% | −0.1 @ 99.9% | −0.5 @ 99.9% |
| Throw-away + short polish | −10.3 @ 88.4% | −0.2 @ 99.9% | −1.0 @ 99.9% |
| Throw-away, pruned layer only | −0.7 @ 98.8% | +0.0 @ 99.9% | not rerun; the skinny pair already matched the group |

Initialization is Kaiming normal, biases zero, batch-norm reset to scale 1 / shift 0. Unit tests check that producer weights change and, in the producers-only scope, consumer weights do not. The empty result is the experiment, not a missed redraw.

Why a dense net tolerated this and a residual CNN does not: the new NEON layer owns its output; a ResNet block is `F(x) + x`, and a random `F` is added to a frozen path. The code also stops a from-scratch group on validation patience 6 from epoch 1 (groups often plateau around epoch 26 of a 60-epoch cap). That is harsher than NEON’s train-loss rule. It is a real caveat. It is not enough to explain an empty band on every net, including the polished arm, because keep-leftover recovers a cut under a *shorter* fine-tune.

Literature that randomly reinitializes CNN weights (Liu, Sun, Wang, Zhang, *Rethinking the Value of Network Pruning*, ICLR 2019) retrains the **entire** pruned architecture for a full training budget. It does not drop one random layer into a frozen residual network. Their guideline, which we should follow: random weights are a fair baseline only when the whole small network is trained. Least-squares refit of the next layer (He et al. 2017; Luo et al., ThiNet) is the published way to adapt consumers while keeping the surviving filters. We have not run that yet.

**Cross off:** throw-away as the CNN training objective; a learned agent under C-G until a non-random replacement recovers the keep-leftover walk.
**Still open for you:** do you want the source’s train-loss patience retried once before that line is final, or is the table enough?

### 2. Benchmark: what was locked, what was measured

Three cells. The agent that we will quote must not have been trained on these architectures.

| Cell | Network | Whose published test |
|---|---|---|
| L1 | CIFAR-10 ResNet-56, DepGraph checkpoint, origin 93.53% | DepGraph Table 1; also the ResNet-56 row of OCS |
| L2 | CIFAR-10 VGG-16-BN, origin 94.16% | OCS |
| L3 | CIFAR-100 VGG-19-BN, DepGraph checkpoint, origin 73.50% | DepGraph Table 1; OCS |

**DepGraph’s own protocol** (Fang et al., CVPR 2023; confirmed on the CVF PDF and on the OCS table that reprints it). They follow ResRep and GReg. The headline CIFAR-10 number is 93.53 → 93.64 (**+0.11**) at **2.57×** fewer FLOPs, with group sparsity learning on that network and then a fine-tune in the style of pretraining (smaller learning rate, fewer iterations). The paper does not print an epoch count in the main text. Their released reproduction script separates a sparsity-learning stage from a fine-tune stage. Without sparsity learning their own ablation is 93.46 (−0.07) at 2.11×. CIFAR-100 VGG-19 is 73.50 → 70.39 (−3.11) at about 8.9×. We will quote these. We will not reimplement the solver. Putting their solver inside SPECTRA would abandon the frozen-agent claim.

**OCS’s own protocol** (Ghimire et al., WACV 2026; CVF PDF). One training cycle from scratch, not prune-then-finetune of a pretrained net. CIFAR: SGD, momentum 0.9, batch 128, **300 epochs**, MultiStep learning-rate schedule. Pruning happens inside the cycle, at an epoch they choose by sub-network stability, and the remaining epochs finish the pruned net. CIFAR numbers are means of three runs. On their ResNet-56 table the row I can read cleanly is **38.88% of FLOPs remaining, 41.42% of parameters, 93.97 → 93.65, drop 0.32**. A second printed row is 38.82 / 42.26, 94.01 → 93.50, drop 0.51. The 21 Sep note used the second. Both are “about 39% of FLOPs and about 42% of parameters.” VGG-16 CIFAR-10 in the same paper keeps as little as 26% or 21% of FLOPs (93.88 and 93.76). VGG-19 CIFAR-100: 70.47 at about 11% of FLOPs. Their DepGraph reprint matches the CVPR table (93.53 → 93.64, drop written as −0.11 because accuracy rose).

**What we measured on our side of that table, recipe A, 2 passes, no agent:**

- Full ResNet-56 twin: mild −3.3 @ 66.1% kept; L1 −5.1 @ 41.5% kept. This is our loop, not a reproduction of their +0.11.
- VGG-16 CIFAR-10: mild −3.5 @ 65.7%; L1 −3.5 @ 41.1%.
- VGG-19 CIFAR-100: the selected point is the **unpruned** network. The walk does leave the band later (validation around −31). We do not quote that as success.

**Learning rate.** Adam at 0.001 is the training recipe. It recovers CIFAR-10 and admitted **0 of 8** CIFAR-100 candidates. Adam at 0.0001 admits 4 of 8 CIFAR-100 nets and fails the CIFAR-10 control by more than a point. SGD at 0.01 admits 2 of 8 and also fails that control. So there is still no one recipe for both datasets. CIFAR-100 stays out of the training catalog until that is decided.

**Learned agents, cheap ResNet-56 only** (not Catalog L):

| Policy | Test Δacc @ params kept |
|---|---|
| In-band linear reward, episode 95, and the same point again at episode 83 | **−7.1 @ 75.6%** |
| Original NEON reward, uncubed (v2c) | −7.4 @ 75.7% |
| Always-keep-90%, 2 passes | −6.6 @ 92.3% |
| Always-keep-90%, 3 passes | −6.9 @ 92.3% |
| Hardest legal cut, 2 passes (L1) | −7.8 @ 89.8% |
| Hardest legal cut, 3 passes | −7.6 @ 91.4% |
| v3 ranking trains and V4 two-decision head, old reward | about −6.7 to −6.9 @ 92% |
| Same actor trained with 40-epoch fine-tune inside learning | −7.1 @ 92.3% (no deeper than the 90% rule) |

The 75.6% point is reproduced. A third pass of either hand rule never selects it. Whether it transfers to Catalog L is the next measurement, and it has not been run: the actor that reached 75.6% was trained on a catalog that already contained the full ResNet-56 and VGG-16, so it cannot be captioned as transfer on L1 or L2. Two newer trains, on a catalog that excludes those architectures, are still running. We will not quote them until a trajectory finishes.

**200-epoch SGD.** One such fine-tune of a single already-pruned ResNet-56 is a few GPU-hours, not a week. Putting 200 epochs inside every step of agent training would multiply the training by roughly an order of magnitude and mix the solver’s budget into the agent. The locked protocol says: not now, and never as a new training job. Optional later, one network, only if a transferred actor is close enough that the fine-tune budget is the remaining question.

### 2b. In flight this week — implemented and submitted on 27 Sep, **no results yet**

These are not findings. They are the experiments that follow from §1 and §2, written here so you see what is running before the numbers arrive. Every one is a switch that is off by default; the running method is unchanged until a test says otherwise.

**Two informed ways to “generate a new layer”, tested without an agent** (same 90 % walk as the table in §1, on the skinny ResNet-56 and on the full-width ResNet-56 twin):

| Recovery | What changes at each cut | What it is compared with | It is dropped if |
|---|---|---|---|
| **A-LSQ** (keep survivors + least-squares refit) | The surviving filters stay. Every layer that *reads* the pruned channels has its weights re-solved in closed form so that it reproduces its own pre-cut output from the channels that remain (He, Zhang, Sun, ICCV 2017). BatchNorm statistics are re-estimated. Then the usual whole-network fine-tune. | today’s recipe at the same size | it is not at least as accurate as today’s recipe at equal size on both networks |
| **C-PCA** (a generated layer, not a random one) | The pruned layer is *replaced* by a layer of the new width whose filters are the principal directions of the old layer’s activations — your “generate a new layer”, with the information kept instead of thrown away. The reading layers are rotated to match; along a residual stream every producer is rotated the same way so the skip additions stay consistent. BatchNorm re-estimated; same fine-tune. | today’s recipe at the same size | it is more than about one point worse than today’s recipe at equal size |

A third small control runs BatchNorm re-estimation alone on today’s recipe, so that any gain of A-LSQ can be attributed to the refit and not to the statistics.

**One fine-tune recipe for every architecture and dataset.** You required that the method not change with the target. Today’s Adam at 0.001 recovers CIFAR-10 and fails CIFAR-100 in 12 epochs; a smaller constant does the opposite. The candidate that could pass both is not a new constant but a **schedule**: AdamW (decoupled weight decay, same decay the SGD arm already had), one epoch of linear warm-up, then cosine decay to 1e-5, inside the same 12 epochs, with BatchNorm re-estimated first. The rationale is Liu et al. (ICLR 2020): Adam’s first steps have a very large variance and warm-up is the standard remedy; our CIFAR-100 failure is an early-step failure. The alternate arm is RAdam, which needs no warm-up. Both run on the CIFAR-10 control and on the eight CIFAR-100 candidates. Pass = within half a point of today’s recipe on CIFAR-10 **and** at least four of eight CIFAR-100 networks recover a real cut. If both arms fail, the training recipe stays Adam 0.001 on CIFAR-10 only and we write that down.

**One new learned agent: cost-shaped actions with an explicit stop.** Instead of “keep 90 % or 80 % of this layer”, the action is “remove 1, 2 or 4 % of the *whole network* through this group”, or “stop here”. The same request then means the same thing on a 16-channel stem and a 256-channel stage, and the agent chooses the operating depth itself instead of our post-hoc rule choosing it. Everything else is the recipe of the best learned policy so far (the linear in-band reward, the same 10-network catalog, the new selection score). It is dropped if its walk still matches a fixed-rate heuristic at equal size on the skinny ResNet-56. The action shape follows AMC (He et al., ECCV 2018), which clipped per-layer sparsity onto a FLOP budget; the explicit stop is ours.

**Fixed this week, not an experiment.** The score that picked which training snapshot to test was the depth of the cut only; it saturated at the depth of the 90 % rule on the two probe networks, so the first snapshot that copied the rule was always the one tested. The new score weights the depth by the accuracy slack that remains, so a kinder policy at the same depth can win.

### 3. Questions for you

1. Is the throw-away table enough to leave random layer replacement as a dense-net method, with keep-leftover as the CNN method — or do you want one retry under the source’s train-loss stopping rule before we close it?
2. For “generate a new layer” on CNNs, is a layer built from the old activations (a projection the next layer can follow) an acceptable reading, as opposed to a random draw?
3. The three bars, in this order: we are cheaper on the next network because there is no per-target search; we try to match our own heuristics at the same size while transferring; we print our short fine-tune next to DepGraph’s +0.11 and expect to be worse. Does that order still match what you want on the slide?

---

## עברית — פירוט

### 1. זריקת המשקולות: מה NEON עשה, ובאיזה שלב

Hirsch & Katz 2022, סעיף 3, בתוך לולאת הגיזום (אלגוריתם 1), לא כשלב נפרד אחרי האימון:

- **החלפת שכבה.** במקום למחוק נוירונים, מייצרים שכבה חדשה בגודל הרצוי. היא מחליפה את השכבה שנבדקה. המשקולות מאותחלות באקראי.
- **כיוונון השכבה.** מקפיאים כל שכבה מלבד החדשה, ומאמנים עד התכנסות.
- אותה לולאה היא גם האימון הלא-מקוון של הסוכן **וגם** שלב המבחן, כשהסוכן המאומן מופעל על רשת שהוא לא התאמן עליה. פעולה «בלי חיתוך» מדלגת גם על ההחלפה וגם על הכיוונון.

בקוד המקורי נבנות גם השכבה הלינארית הבאה וגם נורמליזציית אצווה חדשה, ומאמנים את שלושת המודולים. הסבלנות במקור היא 10 אפוקים על **הפסד האימון**, לא על ולידציה. המשפט במאמר מזכיר רק את השכבה החדשה.

המסלול החי ב-SPECTRA הוא השיטה שהמאמר דחה: שומרים את המסננים ששרדו, ומאמנים את כל הרשת. ניסויי הזריקה על CNN משנים רק את השחזור, על מסלול כפוי של 90%, בלי סוכן.

הטבלה זהה לזו שבחלק האנגלי: השארת המסננים דוחסת; זריקה משאירה כמעט את הרשת המלאה (מעל 98% פרמטרים); ליטוש לא מציל; ציור מחדש של השכבה הגזומה בלבד נכשל כמו הציור המלא.

האתחול הוא Kaiming, ההטיה אפס, והנורמליזציה מאופסת. בדיקות יחידה מאשרות שהמשקולות של השכבה הגזומה משתנות, ובמצב «רק השכבה הגזומה» משקולות השכבה הבאה לא משתנות. התוצאה הריקה היא הניסוי, לא ציור שנשכח.

למה רשת צפופה סבלה את זה ו-ResNet לא: ב-NEON השכבה החדשה בעלת הפלט שלה. ב-ResNet הפלט הוא `F(x) + x`, ו-`F` אקראי מתווסף למסלול קפוא. בנוסף, הקוד עוצר קבוצה מאקראי לפי סבלנות ולידציה של 6 אפוקים מההתחלה (בפועל הרבה קבוצות נעצרות סביב אפוק 26 מתוך 60). זה קשוח יותר מכלל הפסד-האימון של NEON. זו הסתייגות אמיתית. היא לא מסבירה פס ריק על כל הרשתות, כולל זרוע הליטוש, כי השארת המסננים מצליחה בחיתוך תחת כיוונון *קצר יותר*.

הספרות שמאתחלת מחדש משקולות CNN באקראי (Liu ושות׳, ICLR 2019) מאמנת את **כל** הרשת הקטנה לתקציב אימון מלא. היא לא שותלת שכבה אקראית אחת בתוך רשת שיורית קפואה. הכלל שכדאי לאמץ: משקולות אקראיות הן בסיס הוגן רק כשכל הרשת הקטנה מאומנת. התאמה בריבועים פחותים של השכבה הבאה (He ושות׳ 2017; ThiNet) היא הדרך המפורסמת להתאים את הצרכן בלי לזרוק את המסננים ששרדו. את זה עוד לא הרצנו.

**למחוק:** זריקה כיעד אימון ל-CNN; סוכן לומד תחת C-G, עד ששחזור לא-אקראי ישחזר את מסלול השארת-המסננים.
**פתוח אצלך:** האם הטבלה מספיקה כדי להשאיר החלפת שכבה אקראית כשיטה לרשתות צפופות, או שאתה רוצה ניסיון אחד תחת כלל העצירה של המקור לפני שסוגרים.

### 2. מדידה: מה נעול, ומה נמדד

שלושה תאים. הסוכן שנצטט לא יאומן על הארכיטקטורות האלה.

**הפרוטוקול של DepGraph** (Fang ושות׳, CVPR 2023). הם הולכים אחרי ResRep ו-GReg. המספר הראשי ב-CIFAR-10 הוא 93.53 ל-93.64 (**+0.11**) ב-**2.57×** פחות FLOPs, עם למידת דלילות על אותה רשת ואז כיוונון בסגנון האימון המקורי. במאמר עצמו אין מספר אפוקים בטקסט הראשי. בלי למידת דלילות, האבלציה שלהם היא 93.46 (‏0.07−) ב-2.11×. VGG-19 על CIFAR-100: 73.50 ל-70.39 (‏3.11−) בכ-8.9×. נצטט. לא נממש את הפותר. להכניס את הפותר שלהם ל-SPECTRA זה לוותר על הטענה של סוכן קפוא.

**הפרוטוקול של OCS** (Ghimire ושות׳, WACV 2026). מחזור אימון אחד מאפס, לא גיזום-ואז-כיוונון של רשת מאומנת. CIFAR: SGD, מומנטום 0.9, אצווה 128, **300 אפוקים**. הגיזום קורה בתוך המחזור, והאפוקים שנשארים מסיימים את הרשת הגזומה. המספרים הם ממוצע של שלושה ניסיונות. בשורת ResNet-56 שאפשר לקרוא נקי: **38.88% מה-FLOPs נשארים, 41.42% מהפרמטרים, 93.97 ל-93.65, ירידה 0.32**. שורה מודפסת שנייה: 38.82 / 42.26, 94.01 ל-93.50, ירידה 0.51. הפתק מ-21 בספטמבר השתמש בשנייה. שתיהן «בערך 39% FLOPs ובערך 42% פרמטרים».

**מה שנמדד אצלנו, בלי סוכן:** תאום ResNet-56 מוצא חיתוך (מינוס 3.3 ב-66%). VGG-16 על CIFAR-10 דומה. VGG-19 על CIFAR-100: הנקודה שנבחרת היא הרשת המלאה.

**קצב למידה.** Adam ב-0.001 הוא מתכון האימון. הוא משחזר CIFAR-10 והכניס **0 מתוך 8** מועמדי CIFAR-100. Adam ב-0.0001 מכניס 4 מתוך 8 ונכשל בביקורת CIFAR-10 ביותר מנקודה. SGD ב-0.01 מכניס 2 מתוך 8 וגם נכשל בביקורת. אין עדיין מתכון אחד לשני מסדי הנתונים.

**סוכנים שנלמדו, רק על ResNet-56 הזול:** הרשת עם התגמול הלינארי שומרת 75.6% בירידה של 7.1, פעמיים. כלל ה-90% נשאר ב-92% גם במעבר שלישי. הסוכן שהגיע ל-75.6% אומן על קטלוג שכבר כלל את ResNet-56 המלא ואת VGG-16, ולכן אי אפשר לכנות את זה העברה על תאי L1 או L2. שני אימונים חדשים, על קטלוג בלי הארכיטקטורות האלה, עדיין רצים.

**200 אפוקים של SGD.** כיוונון אחד של ResNet-56 שכבר נגזם הוא כמה שעות GPU, לא שבוע. לשים 200 אפוקים בכל צעד של אימון הסוכן מכפיל את האימון בסדר גודל ומערבב את תקציב המאמר לתוך הסוכן. לפי הפרוטוקול שננעל: לא עכשיו, ולא כאימון חדש.

### 2ב. בתנועה השבוע — יושם והוגש ב-27 בספטמבר, **עדיין בלי תוצאות**

אלה לא ממצאים. אלה הניסויים שנובעים מסעיפים 1 ו-2, כתובים כאן כדי שתראה מה רץ לפני שהמספרים מגיעים. כל אחד מהם הוא מתג שכבוי כברירת מחדל; השיטה הרצה לא משתנה עד שמבחן אומר אחרת.

**שתי דרכים «מושכלות» לייצר שכבה חדשה, בלי סוכן** (אותו מסלול 90% כמו בטבלה בסעיף 1, על ResNet-56 הרזה ועל תאום ה-ResNet-56 ברוחב מלא):

| שחזור | מה משתנה בכל חיתוך | מול מה משווים | נזרק אם |
|---|---|---|---|
| **A-LSQ** (שומרים מסננים + התאמת ריבועים פחותים) | המסננים ששרדו נשארים. לכל שכבה ש**קוראת** את הערוצים שנגזמו פותרים מחדש את המשקולות בצורה סגורה, כך שהיא משחזרת את הפלט שלה מלפני החיתוך מתוך הערוצים שנותרו (He, Zhang, Sun, ICCV 2017). סטטיסטיקות ה-BatchNorm נאמדות מחדש. אחר כך הכיוונון הרגיל של כל הרשת. | המתכון של היום באותו גודל | לא לפחות מדויק כמו המתכון של היום באותו גודל, בשתי הרשתות |
| **C-PCA** (שכבה שנוצרת, לא מוגרלת) | השכבה הגזומה **מוחלפת** בשכבה ברוחב החדש שהמסננים שלה הם הכיוונים הראשיים של האקטיבציות של השכבה הישנה — «לייצר שכבה חדשה» שלך, כשהמידע נשמר במקום להיזרק. השכבות הקוראות מסובבות בהתאמה; לאורך זרם שיורי כל היצרנים מסובבים באותו אופן כדי שחיבורי הדילוג יישארו עקביים. BatchNorm נאמד מחדש; אותו כיוונון. | המתכון של היום באותו גודל | גרוע ביותר מנקודה אחת בערך מהמתכון של היום באותו גודל |

בקרה קטנה שלישית מריצה רק את אמידת ה-BatchNorm על המתכון של היום, כדי שרווח של A-LSQ ייוחס להתאמה ולא לסטטיסטיקות.

**מתכון כיוונון אחד לכל ארכיטקטורה ולכל דאטהסט.** דרשת שהשיטה לא תשתנה עם היעד. Adam ב-0.001 של היום משחזר CIFAR-10 ונכשל ב-CIFAR-100 ב-12 אפוקים; קבוע קטן יותר עושה את ההפך. המועמד שיכול לעבור את שניהם אינו קבוע חדש אלא **לוח זמנים**: AdamW (ירידת משקל מנותקת, אותה ירידה שהייתה כבר לזרוע ה-SGD), אפוק אחד של חימום ליניארי, ואז דעיכה קוסינוסית ל-1e-5, בתוך אותם 12 אפוקים, עם אמידת BatchNorm קודם. ההיגיון הוא Liu ושות׳ (ICLR 2020): לצעדים הראשונים של Adam שונות גדולה מאוד וחימום הוא התיקון המקובל; הכישלון שלנו ב-CIFAR-100 הוא כישלון של צעדים ראשונים. הזרוע החלופית היא RAdam, שלא צריך חימום. שתיהן רצות על ביקורת CIFAR-10 ועל שמונה מועמדי CIFAR-100. עובר = בטווח חצי נקודה מהמתכון של היום על CIFAR-10 **וגם** לפחות ארבע מתוך שמונה רשתות CIFAR-100 משחזרות חיתוך אמיתי. אם שתי הזרועות נכשלות, מתכון האימון נשאר Adam 0.001 על CIFAR-10 בלבד, ונכתוב זאת.

**סוכן לומד חדש אחד: פעולות במונחי תקציב עם עצירה מפורשת.** במקום «שמור 90% או 80% מהשכבה הזאת», הפעולה היא «הסר 1, 2 או 4% מ**כל הרשת** דרך הקבוצה הזאת», או «עצור כאן». אותה בקשה אז אומרת אותו דבר על stem של 16 ערוצים ועל שלב של 256 ערוצים, והסוכן בוחר בעצמו את עומק הפעולה במקום שכלל בדיעבד שלנו יבחר אותו. כל השאר הוא המתכון של המדיניות הנלמדת הטובה ביותר עד כה (הגמול הליניארי בתוך הטווח, אותו קטלוג של 10 רשתות, ציון הבחירה החדש). נזרק אם המסלול שלו עדיין תואם היוריסטיקה בקצב קבוע באותו גודל על ResNet-56 הרזה. צורת הפעולה הולכת אחרי AMC (He ושות׳, ECCV 2018), שקטע דלילות פר-שכבה לתקציב FLOPs; העצירה המפורשת היא שלנו.

**תוקן השבוע, לא ניסוי.** הציון שבחר איזה snapshot של האימון נבדוק היה עומק החיתוך בלבד; הוא התרווה בעומק של כלל ה-90% על שתי רשתות הבחינה, ולכן ה-snapshot הראשון שהעתיק את הכלל היה תמיד זה שנבדק. הציון החדש משקלל את העומק ברווח הדיוק שנותר, כך שמדיניות עדינה יותר באותו עומק יכולה לנצח.

### 3. שאלות אליך

1. האם טבלת הזריקה מספיקה כדי להשאיר החלפת שכבה אקראית כשיטה לרשתות צפופות, והשארת מסננים כשיטת ה-CNN — או שאתה רוצה ניסיון אחד תחת כלל העצירה של קוד המקור?
2. האם שכבה שנבנית מהאקטיבציות הישנות (היטל שהשכבה הבאה יכולה לעקוב אחריו) היא קריאה קבילה של «לייצר שכבה חדשה», להבדיל מהגרלה?
3. שלושת הרפים, בסדר הזה: זולים יותר על הרשת הבאה כי אין חיפוש לכל יעד; מנסים להשתוות להיוריסטיקות שלנו באותו גודל תוך העברה; מדפיסים את הכיוונון הקצר שלנו ליד ‎+0.11‎ של DepGraph ומצפים להיות גרועים יותר. האם הסדר הזה עדיין מה שאתה רוצה בשקף?
