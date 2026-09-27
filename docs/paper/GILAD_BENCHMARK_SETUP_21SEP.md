# Note for Dr. Gilad Katz — the benchmarking setup for SPECTRA (reply to your 19 Sep comment)

> **Superseded.** Every point of this note is merged into `docs/paper/GILAD_WEEK_27SEP.md` §1 (the formal status), with the counterparts table, the checkpoint inventory and the measured same-loop rows. The cell labels L1/L2/L3 used here are retired (they collide with L1/L2 pruning): R56·C10 / VGG16·C10 / VGG19·C100. Kept for the record; do not send.

**From:** Ido Paretsky  
**Date:** 21 September 2026  
**What this is:** your comment on the list of networks I sent was right — a list is not an experimental setup. This note gives the setup in the three parts you asked for — (a) what we train on and what we test on, (b) the metrics, (c) the budgets — says what has already been done to make it solid, and what will be done in the coming weeks once we have an agent that beats the simple heuristics by a clear margin. The internal working version, with job numbers and file names, is `docs/paper/CATALOG_L_TEST_PLAN.md`; this note is the readable summary. Hebrew below.

---

## 0. The answer in one paragraph

We will **reproduce the CIFAR test set of a leading, recent structured-pruning paper — DepGraph (Fang et al., CVPR 2023)** — on **their own released checkpoints**: ResNet-56 on CIFAR-10 and VGG-19 on CIFAR-100. We add the CIFAR-10 VGG-16 cell of the newest comparable paper (OCSPruner, WACV 2026), which reports on the same two DepGraph cells and therefore lets us quote both papers on one table. Our agent is **never trained on any of these three architectures**; it is trained once, offline, on a catalog of other CIFAR networks, frozen, and then applied to these three with **no per-network search or training** — the exact opposite of how DepGraph, OCS and AMC work. Every method is reported at **two sizes**: the point our validation rule selects (accuracy drop ≤ 10 points), and the **same size DepGraph/OCS reached** (2.57× fewer FLOPs on ResNet-56), so the numbers sit next to theirs at equal compression. We report **budgets** as measured GPU-hours: ours is one offline training amortised over every later network plus a short fine-tune per network; theirs is a search or a training run *per network*. We will be **more efficient per additional network by construction**, we expect to be **at or above the same-loop heuristics at equal size** (that is the result the next training runs must deliver), and we expect to be **below DepGraph's published accuracy at 2.57×** under our short fine-tune — and we will print that honestly rather than claim otherwise.

---

## 1. (a) What we train on, what we test on

**Training (one offline run, then frozen).** A catalog of pretrained CIFAR networks from four to five families — thin and standard ResNets, VGG-BN, MobileNet-v2, DenseNet-BC — on CIFAR-10, and on CIFAR-100 as soon as our fine-tune recipe recovers CIFAR-100 networks (this is being re-tested this week with a gentler learning rate; the first attempt did not recover them). The catalog deliberately **excludes** every network that appears in any test set below, by architecture, not only by weights: no standard-width ResNet-56, no VGG-16 on CIFAR-10, no VGG-19 on either dataset. Exclusions are enforced by an automatic test that fails if a training file and a test file ever share a network.

**Test set 1 — the reproduced protocol (the committee slide). Three cells, no more:**

| Cell | Network | Checkpoint we prune | Whose test set this is |
|---|---|---|---|
| L1 | CIFAR-10 ResNet-56 | DepGraph's released weights (93.53 %) | DepGraph, OCS, FPGM, Li et al., ResRep, GReg, AMC |
| L2 | CIFAR-10 VGG-16-BN | standard 94.16 % checkpoint | OCS, Network Slimming, Li et al. |
| L3 | CIFAR-100 VGG-19-BN | DepGraph's released weights (73.50 %) | DepGraph, OCS, GReg, PruningBench |

**Test set 2 — transfer coverage (the genericity claim, unchanged).** Networks and datasets the agent never saw: thinner and wider ResNet cousins, VGG-19 on CIFAR-10, DenseNet-100, two families absent from training (ShuffleNet-v2, RepVGG), and two datasets absent from training — **SVHN and Fashion-MNIST** with the same protocol, plus an **ImageNet** ResNet-50 / MobileNet-v2 probe (never trained on, by your instruction). This set answers "did the frozen agent transfer"; the first set answers "how does it look on the field's own benchmark". Neither replaces the other.

**Two operating points per method, on every cell.** (1) *Our rule:* the most compressed point whose **validation** accuracy drop is within 10 points, reported on the **test** set — we never pick the reported point on the test set. (2) *Their size:* the walk continued to DepGraph's compression (2.57× FLOPs on ResNet-56; their ratio on VGG-19; OCS's 42 % parameters on VGG-16), reported even if validation left the band, and labelled as size-matched. Only the second point is printed beside a published number.

**The same loop for every method.** The learned agent and the simple heuristics (keep 90 % of every group; cut 20 % everywhere by L1 magnitude) share the walk, the recovery (40 epochs of fine-tuning, patience 10) and the selection rule. This is the comparison the thesis claim rests on. Published numbers keep their own fine-tuning protocol in the caption; nothing is converted.

## 2. (b) Metrics

One row per method and network:

`original accuracy | pruned accuracy | Δ accuracy (points) | parameters kept (fraction and millions) | FLOPs kept (fraction and millions) | speed-up = 1 / FLOPs kept | fine-tune recipe`

The original accuracy is always printed (our checkpoints and theirs differ by up to a point, and the committee will look for exactly that). Where a heuristic cannot reach the agent's size inside the band, we print that explicitly — the gap between them is the learned-schedule result.

## 3. (c) Budgets — where we are more efficient, and where we are not

| Method | Work done **per target network** | Fine-tune on the target | Cost of the **next** network |
|---|---|---|---|
| **SPECTRA** | **none** — one offline training on the catalog, amortised over every later network | 40 epochs / patience 10 per accepted cut | a single pass over the network plus short fine-tunes; no learning |
| DepGraph | a group-sparsity search on the target | hundreds of SGD epochs (their reproduce script) | repeat search and fine-tune |
| OCSPruner | a one-cycle training on the target | inside the cycle | repeat per network |
| AMC | a reinforcement-learning search per target | then fine-tune | repeat per network |
| L1 / magnitude heuristics | none | our 40 epochs | same as SPECTRA without the agent |

All numbers are **measured**, not estimated: our offline training from the Slurm job time, our per-network cost from one test run, and the compared methods from the epoch counts in their published scripts multiplied by a CIFAR epoch measured on our GPU. We report **two** totals — the cost per additional network (SPECTRA lower by construction) and the cost of the first network including our offline training (SPECTRA higher).

## 4. What "better" will mean — three bars, in this order, never substituted for one another

1. **Budget.** A single frozen agent prunes the field's benchmark networks with zero per-network search. This bar is met by design and will be written first.
2. **Same loop, matched size.** On L1–L3, which the agent never saw in training, its operating point is at least as accurate as the same-loop heuristics at equal size, or reaches a size inside the band that the heuristics cannot. **This is the bar the coming training runs are for.** As of today no agent meets it on the diagnostic networks; the first sign that it can is from this week (below).
3. **Beside the published number, size-matched.** We print our row at 2.57× next to DepGraph's 93.53 → 93.64. We expect to be below it under a 40-epoch Adam fine-tune, and we will say so. The thesis claim is bars 1 and 2 *while transferring*, not bar 3.

**The sentence:** *On the CIFAR test set of DepGraph plus OCS's VGG-16 cell, a single frozen SPECTRA agent — trained once on a catalog containing none of these architectures and never adapted to a target — prunes to a size-matched operating point with no per-network search, at an accuracy we report next to the same-loop heuristics and to the published, differently fine-tuned, results.*

## 5. What has been done to make this solid (as of 21 Sep)

- The protocol above is written and locked as the evaluation chapter's setup; the training catalogs were rebuilt so that no test network appears in them, and an automatic test enforces it (VGG-16 CIFAR-10 was removed from training this week for that reason).
- DepGraph's released ResNet-56 and VGG-19 checkpoints are on our cluster; a loader compatibility check is scheduled before the first run on them. The equivalent standard checkpoints are already runnable and were walked once by the heuristic (ResNet-56: 3.9-point drop at 66 % of parameters kept, inside the band).
- The reward of the learning agent was corrected this week (the in-band term was being compressed by a cube-root; it is now linear). The first agent trained with it is the **first learned policy that cut past the 90 % heuristic on the diagnostic ResNet-56 while staying inside the band** (75.6 % of parameters kept, −7.1 points). It is not yet a win over the heuristics at equal size, but it is the first walk that is not a copy of them.
- An honest accounting of the recovery recipe: your layer-replacement idea was implemented and tested without an agent on three networks (separate note, 19 Sep); on CNNs it did not recover the cut, and keeping the surviving filters did. The evaluation protocol therefore fixes the recovery to the recipe that works.

## 6. What will be done in the coming weeks, in order

1. **Heuristic controls at the agent's sizes** (this week): three-pass walks on the diagnostic pair so that a 75 %-kept point has a heuristic to be compared with; same-loop heuristics on the three L cells.
2. **Loader check of DepGraph's checkpoints**, then the same heuristics on them (no learning involved).
3. **Next agent training** on the cleaned catalog with the corrected reward (queued as soon as a GPU frees), with two changes to how we pick the snapshot we test: the selection score now rewards accuracy inside the band, not only depth of cut (the old score saturated at the heuristic's depth, which is one reason every earlier agent we tested looked like the heuristic), and the probe networks are no longer only thin ResNets.
4. **Tests of that agent** on the coverage set and on L1–L3, at both operating points, printed with the heuristics and the published numbers.
5. **Budget table** filled from measured job times.
6. Optional, only if bar 2 is met and the size-matched ResNet-56 row is close: **one** run with DepGraph's long fine-tuning schedule on that single cell, to see whether the remaining gap is the fine-tune or the schedule.

What we will not do: claim to beat DepGraph or OCS on their home cells; train the agent on ImageNet; put a test network into training to make a table look better.

---

# עברית

**אל:** ד"ר גלעד כץ  
**מ:** עידו פרצקי  
**תאריך:** 21 בספטמבר 2026

ההערה שלך על רשימת הרשתות הייתה נכונה — רשימה אינה experimental setup. במסמך הזה שלושת החלקים שביקשת — (א) על מה מאמנים ועל מה בוחנים, (ב) המטריקות, (ג) התקציבים — ומה כבר נעשה כדי שהמתודולוגיה תעמוד, ומה ייעשה בשבועות הקרובים כשיהיה לנו סוכן שמנצח את ההיוריסטיקות הפשוטות במרווח ברור.

## 0. התשובה בפסקה אחת

נשחזר את **סט הבדיקה של CIFAR של מאמר מוביל ועדכני בגזימה מבנית — DepGraph (Fang et al., CVPR 2023)** — על **המשקלים שהם עצמם שחררו**: ResNet-56 על CIFAR-10 ו-VGG-19 על CIFAR-100. נוסיף את התא VGG-16 על CIFAR-10 מהמאמר העדכני ביותר בתחום (OCSPruner, WACV 2026), שמדווח בדיוק על שני התאים של DepGraph, ולכן אפשר לצטט את שני המאמרים בטבלה אחת. הסוכן שלנו **לעולם לא מתאמן על אף אחת משלוש הארכיטקטורות האלה**: הוא מתאמן פעם אחת, אופליין, על קטלוג של רשתות CIFAR אחרות, מוקפא, ומופעל על השלוש **בלי חיפוש או אימון פר-רשת** — ההפך הגמור מאיך ש-DepGraph, OCS ו-AMC עובדים. כל שיטה מדווחת ב**שני גדלים**: הנקודה שכלל הולידציה שלנו בוחר (ירידת דיוק של עד 10 נקודות), ו**הגודל שאליו DepGraph/OCS הגיעו** (פי 2.57 פחות FLOPs על ResNet-56), כך שהמספרים יושבים ליד שלהם באותה דחיסה. את **התקציבים** נדווח כשעות GPU מדודות: אצלנו אימון אופליין אחד שמתפרס על כל רשת עתידית ועוד כיול קצר לכל רשת; אצלם חיפוש או אימון *לכל רשת*. נהיה **יעילים יותר לכל רשת נוספת מעצם הבנייה**; אנחנו מצפים להיות **לפחות ברמת ההיוריסטיקות באותו גודל** (זו התוצאה שאימוני ההמשך צריכים לספק); ואנחנו מצפים להיות **מתחת לדיוק שפורסם ב-DepGraph בפי 2.57** תחת הכיול הקצר שלנו — ונכתוב את זה ביושר ולא נטען אחרת.

## 1. (א) על מה מאמנים, על מה בוחנים

**אימון (ריצה אופליין אחת, ואז הקפאה).** קטלוג של רשתות CIFAR מאומנות מארבע-חמש משפחות — ResNet רזה וסטנדרטי, VGG-BN, MobileNet-v2, DenseNet-BC — על CIFAR-10, ועל CIFAR-100 ברגע שמתכון הכיול שלנו ישחזר רשתות CIFAR-100 (נבדק מחדש השבוע עם קצב למידה עדין יותר; הניסיון הראשון לא שחזר אותן). הקטלוג **מוציא** בכוונה כל רשת שמופיעה בסטי הבדיקה, לפי ארכיטקטורה ולא רק לפי משקלים: אין ResNet-56 ברוחב סטנדרטי, אין VGG-16 על CIFAR-10, אין VGG-19 על אף אחד מהדאטהסטים. ההוצאות נאכפות בטסט אוטומטי שנכשל אם קובץ אימון וקובץ בדיקה חולקים רשת.

**סט בדיקה 1 — הפרוטוקול המשוחזר (שקף הוועדה). שלושה תאים, לא יותר:** L1 — ResNet-56 על CIFAR-10 במשקלים של DepGraph (93.53%); L2 — VGG-16-BN על CIFAR-10 (94.16%); L3 — VGG-19-BN על CIFAR-100 במשקלים של DepGraph (73.50%).

**סט בדיקה 2 — כיסוי העברה (טענת הגנריות, ללא שינוי).** רשתות ודאטהסטים שהסוכן לא ראה: בני-דודים של ResNet רזים ורחבים יותר, VGG-19 על CIFAR-10, DenseNet-100, שתי משפחות שאינן באימון (ShuffleNet-v2, RepVGG), ושני דאטהסטים שאינם באימון — **SVHN ו-Fashion-MNIST** באותו פרוטוקול, ועוד בדיקת **ImageNet** על ResNet-50 / MobileNet-v2 (בלי אימון, לפי הנחייתך). הסט הזה עונה "האם הסוכן המוקפא עבר"; הראשון עונה "איך זה נראה על הבנצ'מרק של התחום". אף אחד לא מחליף את השני.

**שתי נקודות פעולה לכל שיטה, בכל תא.** (1) *הכלל שלנו:* הנקודה הדחוסה ביותר שבה ירידת הדיוק על **ולידציה** היא עד 10 נקודות, מדווחת על **סט הבדיקה** — לעולם לא בוחרים את הנקודה על סט הבדיקה. (2) *הגודל שלהם:* ממשיכים את ההליכה עד לדחיסה של DepGraph (פי 2.57 ב-FLOPs על ResNet-56; היחס שלהם על VGG-19; 42% פרמטרים של OCS על VGG-16), מדווחים גם אם הולידציה יצאה מהטווח, ומסמנים "מותאם-גודל". רק הנקודה השנייה מודפסת ליד מספר שפורסם.

**אותה לולאה לכל שיטה.** הסוכן וההיוריסטיקות הפשוטות (לשמור 90% מכל קבוצה; לגזור 20% בכל מקום לפי נורמת L1) חולקים את ההליכה, את השחזור (40 אפוקים של כיול, סבלנות 10) ואת כלל הבחירה. זו ההשוואה שעליה נשענת טענת התזה. מספרים שפורסמו נשארים עם פרוטוקול הכיול שלהם בכיתוב; לא ממירים דבר.

## 2. (ב) מטריקות

שורה אחת לכל שיטה ורשת: `דיוק מקורי | דיוק אחרי גזימה | Δדיוק (נקודות) | פרמטרים שנשמרו (שבר ומיליונים) | FLOPs שנשמרו (שבר ומיליונים) | האצה = 1 / FLOPs שנשמרו | מתכון הכיול`. הדיוק המקורי מודפס תמיד (המשקלים שלנו ושלהם נבדלים בעד נקודה, ובדיוק שם הוועדה תחפש). כשההיוריסטיקה לא מגיעה לגודל של הסוכן בתוך הטווח — כותבים זאת במפורש; הפער הזה הוא תוצאת "לוח הזמנים הנלמד".

## 3. (ג) תקציבים — היכן אנחנו יעילים יותר, והיכן לא

SPECTRA: **אפס** עבודה פר-רשת — אימון אופליין אחד על הקטלוג, מתפרס על כל רשת עתידית; כיול של 40 אפוקים לכל גזירה שהתקבלה; הרשת הבאה עולה מעבר אחד ועוד כיולים קצרים, בלי למידה. DepGraph: חיפוש group-sparsity על הרשת + מאות אפוקי SGD (סקריפט השחזור שלהם), חוזר לכל רשת. OCSPruner: אימון one-cycle על הרשת, חוזר לכל רשת. AMC: חיפוש RL לכל רשת ואז כיול. היוריסטיקות L1: בלי חיפוש, 40 אפוקים שלנו — כמו SPECTRA בלי הסוכן. כל המספרים **נמדדים**: האימון האופליין מזמן הג'וב ב-Slurm, העלות פר-רשת מריצת בדיקה אחת, והשיטות המושוות ממספר האפוקים בסקריפטים שפורסמו כפול אפוק CIFAR שנמדד על ה-GPU שלנו. מדווחים **שני** סכומים — העלות לרשת נוספת (SPECTRA נמוכה יותר מעצם הבנייה) והעלות של הרשת הראשונה כולל האימון האופליין (SPECTRA גבוהה יותר).

## 4. מה יהיה פירוש "טובים יותר" — שלושה רפים, בסדר הזה, בלי להחליף ביניהם

1. **תקציב.** סוכן מוקפא אחד גוזם את רשתות הבנצ'מרק של התחום בלי חיפוש פר-רשת. הרף הזה מתקיים מעצם התכנון.
2. **אותה לולאה, גודל תואם.** על L1–L3, שהסוכן לא ראה באימון, נקודת הפעולה שלו מדויקת לפחות כמו ההיוריסטיקות באותו גודל, או מגיעה לגודל בתוך הטווח שההיוריסטיקות לא מגיעות אליו. **זה הרף שלמענו אימוני ההמשך.** נכון להיום אף סוכן לא עומד בו על רשתות האבחון; הסימן הראשון שהוא יכול הגיע השבוע (למטה).
3. **ליד המספר שפורסם, בגודל תואם.** נדפיס את השורה שלנו בפי 2.57 ליד 93.53→93.64 של DepGraph. אנחנו מצפים להיות מתחת תחת כיול Adam של 40 אפוקים, ונאמר זאת. טענת התזה היא רפים 1 ו-2 *תוך העברה*, לא רף 3.

## 5. מה כבר נעשה (נכון ל-21 בספטמבר)

- הפרוטוקול נכתב וננעל כ-setup של פרק ההערכה; קטלוגי האימון נבנו מחדש כך שאף רשת בדיקה לא מופיעה בהם, וטסט אוטומטי אוכף זאת (VGG-16 על CIFAR-10 הוצא מהאימון השבוע מסיבה זו).
- המשקלים של DepGraph ל-ResNet-56 ול-VGG-19 נמצאים על הקלאסטר; בדיקת תאימות טעינה מתוזמנת לפני הריצה הראשונה עליהם. המשקלים הסטנדרטיים המקבילים כבר רצים, וההיוריסטיקה הלכה עליהם פעם אחת (ResNet-56: ירידה של 3.9 נקודות ב-66% מהפרמטרים, בתוך הטווח).
- פונקציית הגמול של הסוכן תוקנה השבוע (הזרוע שבתוך הטווח נדחסה בשורש שלישי; כעת היא ליניארית). הסוכן הראשון שאומן איתה הוא **המדיניות הנלמדת הראשונה שגזרה מעבר להיוריסטיקת ה-90% על ResNet-56 האבחוני ונשארה בתוך הטווח** (75.6% מהפרמטרים, −7.1 נקודות). זה עוד לא ניצחון על ההיוריסטיקות באותו גודל, אבל זו ההליכה הראשונה שאינה העתק שלהן.
- חשבון כנה על מתכון השחזור: רעיון החלפת השכבה שלך יושם ונבדק בלי סוכן על שלוש רשתות (הערה נפרדת, 19 בספטמבר); על CNN הוא לא שחזר את הגזירה, ושמירת הפילטרים ששרדו כן. לכן פרוטוקול ההערכה מקבע את השחזור למתכון שעובד.

## 6. מה ייעשה בשבועות הקרובים, לפי הסדר

1. **בקרות היוריסטיות בגדלים של הסוכן** (השבוע): הליכות של שלושה מעברים על זוג האבחון, כדי שלנקודת 75% תהיה היוריסטיקה להשוואה; היוריסטיקות באותה לולאה על שלושת תאי L.
2. **בדיקת טעינה של המשקלים של DepGraph**, ואז אותן היוריסטיקות עליהם (בלי למידה).
3. **אימון הסוכן הבא** על הקטלוג המנוקה עם הגמול המתוקן (בתור לרגע שמתפנה GPU), עם שני שינויים באיך בוחרים את ה-snapshot שנבדוק: ציון הבחירה מתגמל כעת דיוק בתוך הטווח ולא רק עומק גזירה (הציון הישן התרווה בעומק של ההיוריסטיקה — אחת הסיבות שכל סוכן שבדקנו נראה כמו ההיוריסטיקה), ורשתות הבחינה אינן עוד רק ResNet רזות.
4. **בדיקות של אותו סוכן** על סט הכיסוי ועל L1–L3, בשתי נקודות הפעולה, לצד ההיוריסטיקות והמספרים שפורסמו.
5. **טבלת התקציבים** ממולאת מזמני ג'ובים מדודים.
6. אופציונלי, רק אם רף 2 מתקיים ושורת ResNet-56 המותאמת-גודל קרובה: **ריצה אחת** עם לוח הכיול הארוך של DepGraph על התא הבודד הזה, כדי לראות אם הפער שנותר הוא הכיול או לוח הזמנים.

מה שלא נעשה: לא נטען שניצחנו את DepGraph או OCS על התאים הביתיים שלהם; לא נאמן את הסוכן על ImageNet; לא נכניס רשת בדיקה לאימון כדי שטבלה תיראה טוב יותר.
