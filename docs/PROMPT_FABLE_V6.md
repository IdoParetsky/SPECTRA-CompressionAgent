# SPECTRA V6 sitting — Fable 5.1 (19 Sep 2026, ~16:15 IDT)

**OPS DELTA 27 Sep 02:46 IDT — 3-hour briefing. PPO-8 finished on patience. The saved score did not move.**

- **Good.** PPO-8 `21536397` finished at 02:27, exit 0, after 4 days 15 hours. The stop line is `reward_not_improving`, `since_improvement=152/150`, `min_episodes=300`, `rewinds=3`, elapsed `400521/518400`. That is patience after the 300-episode floor, with time left on the six-day clock. No traceback. Factored `21536398` is still running. Its best is still **0.061**. PPO-8’s critic on the last learning step was **0.98**, and its best stayed **0.067**. The tested point remains episode 95, **−7.1 @ 0.756**.
- **Bad.** The last PPO-8 probe is still episode 288 at **0.065** (ResNet-56 **0.029**, ResNet-20 **0.101**), under the freeze at **0.068**. Probe 304 never ran. No new snapshot. Factored’s probe is still episode 228 at **0.060** (ResNet-56 **0.031**), under **0.061** by **0.0006**. Its critic fell from **0.88** to **0.54** on update 60, with clip fraction **0.00**. QOS is 1/6. Five GPUs are empty.
- **Insight.** This is the same ending as the area train: the minimum episode count, then patience, and the first freeze still the best. Reusing each batch for 8 epochs did not write a higher area score. A critic of **0.98** sat on that stop. Factored’s critic has now dipped twice (to **0.60**, then to **0.54**) while its saved best stayed **0.061**. The critic fit is not the selection score.
- **Fable additions.** (1) Do not TEST PPO-8 episode 143 or 288. The train has ended. The freeze stays **0.068**. (2) Do not TEST factored episode 228 or 167, or area episode 83 or 240. (3) Do not scancel the factored train for the critic. (4) Sample reuse is neither confirmed nor crossed off: the kill rule needs a trajectory, and this probe did not beat the freeze. (5) Five GPUs are free. Ops will not invent a cell. The next GPU work is A-LSQ after `docs/PROMPT_FABLE_NEXT_SITTING.md`, not another agent. (6) One fine-tune recipe is still the first decision in that prompt. (7) The 09:30 board should show PPO-8 finished. The 23:00 board still shows it running. (8) The extension report still waits on your format.

**OPS DELTA 26 Sep 23:46 IDT — 3-hour briefing. Both pending probes landed. Neither replaced its freeze.**

- **Good.** Two jobs are running, no tracebacks. QOS is 2/6. PPO-8’s probe at episode 288 is **0.065** (ResNet-56 **0.029**, ResNet-20 **0.101**), up from **0.050** at episode 272. Factored’s probe at episode 228 is **0.060** (ResNet-56 **0.031**, ResNet-20 **0.089**), up from **0.027** at episode 216. Factored’s critic rose from **0.60** to **0.74** on update 58, and the clip fraction fell from **0.22** to **0.04**. PPO-8’s critic is **0.96**, and its best is still **0.067**. Factored’s best is still **0.061**. The tested point remains episode 95, **−7.1 @ 0.756**. The 23:00 board already shows these two probes.
- **Bad.** **0.065** is under the PPO-8 freeze at **0.068**. **0.060** is under the factored freeze at **0.061**, by **0.0006**. No new snapshot on either job. The ResNet-56 halves are **0.029** and **0.031**, against **0.167** on the finished in-band actor. Four GPUs are empty. The 20:45 note is stale: it still quotes PPO-8 at **0.050** and factored at **0.027**.
- **Insight.** The two probes the 20:45 note was waiting on both rose, and both stopped short of the freeze. Factored walked back to its own freeze on both nets and stayed **0.0006** under. The critic recovered on that same stretch. A higher critic is not a new snapshot. PPO-8’s ResNet-56 half went from **0.023** to **0.029**, still under the freeze’s own ResNet-56 half of **0.031**.
- **Fable additions.** (1) Do not TEST PPO-8 episode 288 or 143, factored episode 228 or 167, or area episode 83 or 240. (2) Do not scancel the factored train. The critic dip reversed. (3) Do not compare these area-kind scores with the structural probe near 0.26. (4) **0.060** under **0.061** is not a tie and not a new freeze. (5) Four GPUs are free. Ops will not invent a cell. A longer fine-tune, if you name it, is one frozen-actor test of episode 95, outside a new training job. (6) The extension report still waits on your format. (7) The next PPO-8 probe is episode 304. The next factored probe is episode 240.

**OPS DELTA 26 Sep 20:45 IDT — 3-hour briefing. No new probe. Factored critic fell again.**

- **Good.** Two jobs are running, no tracebacks. QOS is 2/6. PPO-8’s critic is **0.95**, and its best is still **0.067**. Factored’s best is still **0.061**. The probes stand where 17:45 left them: PPO-8 episode 272 at **0.050** (ResNet-56 **0.023**), factored episode 216 at **0.027** (ResNet-56 **0.025**, ResNet-20 **0.028**). The tested point remains episode 95, **−7.1 @ 0.756**.
- **Bad.** Neither job printed a new probe. Both scores stay under their freezes, **0.068** and **0.061**. Factored’s critic fell from **0.74** to **0.60** on update 57, with clip fraction **0.22**. Four GPUs are empty.
- **Insight.** Three hours, including a ResNet-56 training episode on the factored job, did not change the selection score. The best and the freeze stayed put while the critic fell. The next factored probe is episode 228. The next PPO-8 probe is episode 288.
- **Fable additions.** (1) Do not TEST PPO-8 episode 272 or 143, factored episode 216 or 167, or area episode 83 or 240. (2) Do not scancel the factored train for the critic. (3) Do not compare these area-kind scores with the structural probe near 0.26. (4) Four GPUs are free. Ops will not invent a cell. A longer fine-tune, if you name it, is one frozen-actor test of episode 95, outside a new training job. (5) The extension report still waits on your format. (6) The 23:00 board should show PPO-8 **0.050** and factored **0.027**. The 16:15 board still shows factored at **0.011**.

**OPS DELTA 26 Sep 17:45 IDT — 3-hour briefing. The two probes moved in opposite directions.**

- **Good.** Two jobs are running, no tracebacks. QOS is 2/6. Factored’s probe at episode 216 is **0.027** (ResNet-56 **0.025**, ResNet-20 **0.028**), up from **0.011** at episode 204. PPO-8’s critic is **0.95**, and its best is still **0.067**. Factored’s best is still **0.061**. The tested point remains episode 95, **−7.1 @ 0.756**.
- **Bad.** PPO-8’s probe at episode 272 is **0.050** (ResNet-56 **0.023**, ResNet-20 **0.076**), down from **0.064**, under the freeze at **0.068**. Factored’s **0.027** is under the freeze at **0.061**. No new snapshot on either job. Factored’s critic fell to **0.74** on update 55, with clip fraction **0.28**. Four GPUs are empty.
- **Insight.** Both probes moved, and neither replaced its freeze. Factored’s rise is on both nets, and it is still under half of **0.061**. PPO-8’s ResNet-56 half went from **0.031** back to **0.023**. A critic of **0.95** sat on that drop.
- **Fable additions.** (1) Do not TEST PPO-8 episode 272 or 143, factored episode 216 or 167, or area episode 83 or 240. (2) Do not scancel the factored train for the critic dip. (3) Do not compare these area-kind scores with the structural probe near 0.26. (4) Four GPUs are free. Ops will not invent a cell. A longer fine-tune, if you name it, is one frozen-actor test of episode 95, outside a new training job. (5) The extension report still waits on your format. (6) The 23:00 board should add factored episode 216 at **0.027**. The 16:15 board has PPO-8 episode 272 and still shows factored at **0.011**.

**OPS DELTA 26 Sep 14:44 IDT — 3-hour briefing. Both probes held. The episode counters moved.**

- **Good.** Two jobs are running, no tracebacks. QOS is 2/6. PPO-8’s probe is still episode 256 at **0.064** (ResNet-56 **0.031**, ResNet-20 **0.097**). The factored critic is **0.84**, up from **0.78** at update 52, and its best is still **0.061**. The tested point remains episode 95, **−7.1 @ 0.756**.
- **Bad.** Neither job printed a new probe. PPO-8’s **0.064** is still under the freeze at **0.068**, and the critic is **0.87**. Factored’s probe is still episode 204 at **0.011** (ResNet-56 **0.012**, ResNet-20 **0.010**), under the freeze at **0.061**. PPO-8’s last logged episode is 268 at 14:10. The area train stays finished, best episode 83 at **0.059**. Four GPUs are empty.
- **Insight.** Three hours advanced the episode counters and left the selection scores where the 11:45 poll put them. The next factored probe is episode 216. The next PPO-8 probe is episode 272.
- **Fable additions.** (1) Do not TEST PPO-8 episode 256 or 143, factored episode 204 or 167, or area episode 83 or 240. (2) Do not scancel the factored train for the critic, or PPO-8 for the dip to **0.87**. (3) Do not compare these area-kind scores with the structural probe near 0.26. (4) Four GPUs are free. Ops will not invent a cell. A longer fine-tune, if you name it, is one frozen-actor test of episode 95, outside a new training job. (5) The extension report still waits on your format. (6) The 16:00 board should replace the 09:43 stamp. It does not yet have the 0.064 and 0.011 probes.

**OPS DELTA 26 Sep 11:45 IDT — catch-up after the VPN gap. PPO-8 ResNet-56 half came back. Factored stayed down.**

- **Good.** Login returned at 11:45. Two jobs are running, no tracebacks. QOS is 2/6. PPO-8’s probe at episode 256 is **0.064** (ResNet-56 **0.031**, ResNet-20 **0.097**). That ResNet-56 half was **0.002** at episode 240. The factored critic is **0.92**, and its best is still **0.061**. The tested point remains episode 95, **−7.1 @ 0.756**.
- **Bad.** PPO-8’s **0.064** is under the freeze at **0.068**. No new snapshot. Factored’s probe at episode 204 is **0.011** (ResNet-56 **0.012**, ResNet-20 **0.010**), under the freeze at **0.061**, and level with the **0.012** probe at episode 192. The area train finished at 09:21, exit 0, after 250 episodes, because the reward had stopped improving. Its best stays episode 83 at **0.059**. Its last probe is **0.053**, ResNet-56 half **0.028**. Four GPUs are empty.
- **Insight.** The PPO-8 ResNet-56 collapse was one probe. Episode 256 put that half back on the freeze’s own ResNet-56 half, and the combined score still did not replace the snapshot. Factored has now scored about **0.01** on both nets at two probes in a row. The area train ended on the same pattern as the finished in-band train: patience after episode 250, best unchanged.
- **Fable additions.** (1) Do not TEST PPO-8 episode 256 or 143, factored episode 204 or 167, or area episode 83 or 240. (2) Do not scancel the factored train. (3) Do not compare these area-kind scores with the structural probe near 0.26. (4) Four GPUs are free. Ops will not invent a cell. A longer fine-tune, if you name it, is one frozen-actor test of episode 95, outside a new training job. (5) The extension report still waits on your format. (6) The 09:43 board does not have these two probes. The next board slot is 16:00.

**OPS DELTA 26 Sep 09:13 IDT — 3-hour briefing. Factored probe fell. The other two held.**

- **Good.** Login is up. Three jobs are running, no tracebacks. QOS is 3/6. The tested point remains episode 95, **−7.1 @ 0.756**. The factored critic is **0.94**. PPO-8’s critic is **0.95**. Both long trains stay finished: in-band tied its freeze, and the 40-epoch train stopped on the six-day clock.
- **Bad.** Factored’s probe at episode 192 is **0.012** (ResNet-56 **0.010**, ResNet-20 **0.014**), down from **0.045** at episode 180, under the freeze at **0.061**. Area’s probe is still **0.053** (ResNet-56 **0.028**), under **0.059**, and the critic is **0.68**. PPO-8’s probe is still **0.048**, ResNet-56 half **0.002**, under **0.068**. No new snapshot. Three GPUs are empty.
- **Insight.** Since 06:12 the only probe that moved is the factored one, and it moved down on both nets. A critic of **0.94** sat on top of that drop. The live ResNet-56 halves are **0.028**, **0.002**, and **0.010**, against **0.167** on the finished in-band train.
- **Fable additions.** (1) Do not TEST factored episode 167 or 192, area episode 240, or PPO-8 episode 240. (2) Do not scancel the factored train for the probe drop, or the area train for the critic. (3) Do not compare these area-kind scores with the structural probe near 0.26. (4) Three GPUs are free. Ops will not invent a cell. A longer fine-tune, if you name it, is one frozen-actor test of episode 95, outside a new training job. (5) The extension report still waits on your format. (6) The 09:30 board replaces the stale 23:00 stamp.

**OPS DELTA 26 Sep 06:12 IDT — 3-hour briefing. Area probe ticked up. ResNet-56 did not.**

- **Good.** Three jobs are still running, no tracebacks. The area probe at episode 240 is **0.053**, up from **0.049**. The factored critic is **0.89**, and its best is still **0.061**. PPO-8’s critic is **0.94**. QOS is 3/6.
- **Bad.** **0.053** is still under the area freeze at **0.059**, and the ResNet-56 half is **0.028**. PPO-8’s probe is still **0.048**, and its ResNet-56 half is **0.002**, under the freeze at **0.068**. Factored’s probe is still **0.045**, under its first freeze at **0.061**. No new snapshot on any of the three. The area critic fell from **0.93** to **0.72** at update 60. The tested points remain episode 95 and episode 59.
- **Insight.** Three hours moved one area-kind probe by four thousandths. The ResNet-56 half of that probe stayed at **0.028**. PPO-8’s ResNet-56 half at **0.002** is the live policy walking away from the freeze, and the freeze itself stays episode 143.
- **Fable additions.** (1) Do not TEST area episode 240, PPO-8 episode 240, or factored episode 180. (2) Do not scancel the area train for the critic dip, or the factored train. (3) Do not compare these area-kind scores with the structural probe near 0.26. (4) Three GPUs are free. Ops will not invent a cell. (5) The 09:30 board should replace the stale 23:00 stamp. (6) The extension report still waits on Ido’s format.

**OPS DELTA 26 Sep 02:45 IDT — 3-hour briefing. Login returned. Both long trains have ended.**

- **Good.** The 02:43 poll reached Slurm. Three jobs are running, no tracebacks. The factored train froze for the first time: episode 167, area-kind **0.061** (ResNet-56 half **0.034**, ResNet-20 half **0.088**). Area’s critic is **0.89**. PPO-8’s critic is **0.96**, and its freeze at **0.068** is still in place. The tested in-band point remains episode 95, **−7.1 @ 0.756**.
- **Bad.** The 40-epoch train finished at 07:17 on 25 Sep, exit 0, because the six-day clock ran out. The reward had not stopped improving (`104/150`). Its best freeze is still episode 59. Its last probe is **0.225**, ResNet-56 half **0.090**. Area’s probe at episode 228 is **0.049**, under **0.059**. PPO-8’s probe at episode 224 is **0.052**, under **0.068**. Factored’s next probe, episode 180, is **0.045**, under the new freeze, and the critic is **0.54** with clip fraction **0.47**. QOS is 3/6.
- **Insight.** The vacation gap hid two clean endings and one small first freeze. Neither ending beat its tested snapshot. The three live ResNet-56 halves are about **0.03**. An area-kind **0.061** is a different scale from the structural probe near 0.26.
- **Fable additions.** (1) Do not TEST the factored episode 167, the PPO-8 freeze, the area probe, the 40-epoch episode 156, or in-band episode 240. The tested points remain episode 95 (§123) and episode 59 (§112). (2) Do not scancel the factored train. (3) Do not start a 200-epoch rematch inside a training job. A longer fine-tune, if Ido names it, is one frozen-actor test of episode 95. (4) Three GPUs are free. Ops will not invent a cell. (5) The extension report still waits on Ido’s format.

**OPS DELTA 25 Sep 06:18 IDT — 3-hour briefing. Still no reading since 02:19.**

- **Good.** The sleeper is running. The last poll that reached Slurm, at 02:19, had four jobs running and no tracebacks. Those jobs do not need this laptop. The scores at that poll stand: 40-epoch probe **0.225** (ResNet-56 half **0.090**, ResNet-20 half **0.360**), area **0.043** with critic **0.90** and best **0.059**, PPO-8 probe **0.063** under its freeze at **0.068**, factored **0.028** with best **0.049** and no freeze. The in-band train remains finished, exit 0, last probe **0.262** tied with episode 95.
- **Bad.** `slurm.bgu.ac.il` has failed on every wake since 02:47, including this one, twice. At 05:48 the name did not resolve, then the retry timed out. There is no reading for probes that could have landed after 02:19 (40-epoch 168, area 168, PPO-8 172, factored 132). A failed job in that window would not show up here.
- **Insight.** This is still the laptop path to the login node. Four hours of silence leaves the ranking where the 02:19 poll left it. The best tested actor remains the finished in-band train, episode 95, **−7.1 @ 0.756** (§123). Every ResNet-56 half at that poll was unchanged: **0.167**, **0.090**, **0.026**, **0.034**, and **0.026**.
- **Fable additions.** (1) Do not invent a score after 02:19. The first poll that reaches the login is a catch-up. (2) Do not TEST from this gap. The tested points remain in-band episode 95 (§123) and 40-epoch episode 59 (§112). (3) Do not scancel anything from the 02:19 poll. (4) Do not compare the area-kind scores with the structural probe near 0.26. (5) Two GPUs were free at 02:19. Ops will not invent a cell. (6) The 23:00 board still shows the in-band train as live. The 09:30 stamp replaces it, and if the login is still down that stamp stays stale. (7) The extension report still waits on Ido’s format.

**OPS DELTA 25 Sep 03:17 IDT — 3-hour briefing. No reading since 02:19.**

- **Good.** At 02:19 four jobs were running, no tracebacks. The area critic had recovered from **0.65** to **0.90**, and its best was still **0.059**. The 40-epoch probe at episode 156 was **0.225**, up from **0.180**, on the ResNet-20 half (**0.360**). PPO-8’s freeze at **0.068** was still in place, with critic **0.94**. The in-band train remains finished, exit 0. The sleeper is running.
- **Bad.** Port 22 timed out at 02:47 and again on this wake, twice. There is no reading after 02:19. The 40-epoch ResNet-56 half is still **0.090**, under the episode-59 freeze at **0.268**. Area’s probe is still **0.043**, under **0.059**. PPO-8’s probe at episode 160 is **0.063**, under its new freeze, and the ResNet-56 half of that freeze is **0.031**. Factored’s probe is still **0.028**, its best is still **0.049**, its critic is **0.63**, and there is still no freeze. Probes that could have landed after 02:19 (PPO-8 172, factored 132) have not been seen.
- **Insight.** Three hours after the catch-up, the only combined score that rose was the 40-epoch probe, and the rise is ResNet-20. Every ResNet-56 half is where the 00:19 poll left it: **0.167** on the finished in-band train, **0.090** on the 40-epoch train, **0.026**, **0.034**, and **0.026** on the three new trains. The quoted test remains episode 95, **−7.1 @ 0.756** (§123).
- **Fable additions.** (1) Do not invent a score after 02:19. The first poll that reaches the login is a catch-up. (2) Do not TEST the 40-epoch episode 156, area episode 156, PPO-8 episode 143, or factored episode 120. The tested points remain in-band episode 95 (§123) and 40-epoch episode 59 (§112). (3) Do not scancel the factored train for the critic, or the area train. (4) Do not compare these area-kind scores with the structural probe near 0.26. (5) Two GPUs are free. Ops will not invent a cell. (6) The 23:00 board still shows the in-band train as live. The 09:30 stamp replaces it. (7) The extension report still waits on Ido’s format.

**OPS DELTA 25 Sep 00:19 IDT — 3-hour briefing. Login returned. In-band train has ended.**

- **Good.** The 00:19 poll reached Slurm. Four jobs are running, no tracebacks. The in-band train **21459737** finished at 19:04, exit 0, after 5 days 1 hour. The stop line is `reward_not_improving`, `since_improvement=152/150`, `min_episodes=250`, `rewinds=3`. PPO update 62 has critic **0.68**, best still **0.262**, clip fraction **0.012**. PPO-8 froze a new snapshot at 16:42: episode 143, area-kind score **0.068** (ResNet-56 half **0.031**, ResNet-20 half **0.104**), above its old freeze at **0.063**. Its critic at update 20 is **0.95**.
- **Bad.** That in-band stop did not beat episode 95. The last probe is still episode 240 at **0.262**, the ResNet-56 half is still **0.167**, and there is no new snapshot. Probe 252 never ran. Area’s probes are **0.055** at episode 144 and **0.043** at episode 156, both under the **0.059** freeze, and the critic fell from **0.92** to **0.65** at update 40. PPO-8’s next probe, episode 160, is **0.063**, under the new freeze. Factored’s probe fell from **0.046** at episode 108 to **0.028** at episode 120. Its best is still **0.049**, and there is still no freeze. The 40-epoch train is on episode 155. Its last probe is still episode 144 at **0.180**. QOS is 4/6.
- **Insight.** The train that holds the best tested point has finished. It stopped because the reward stopped improving, with the selection score tied to the freeze. PPO-8’s new freeze is an area-kind **0.068**, and the gain is the ResNet-20 half. The ResNet-56 half is **0.031**. The blind spot hid a clean stop and one small freeze. The quoted test remains episode 95, **−7.1 @ 0.756** (§123).
- **Fable additions.** (1) Do not TEST in-band episode 240. The train has ended. The tested point remains episode 95 (§123). (2) Do not TEST PPO-8 episode 143. An area-kind **0.068** is a different scale from the structural probe near 0.26. (3) Do not TEST area episode 156, factored episode 120, or the 40-epoch episode 144. The tested 40-epoch point remains episode 59 (§112). (4) Do not scancel the area train for the critic dip, and do not scancel the factored train. (5) Two GPUs are free. Ops will not invent a cell. (6) The 23:00 board still shows the in-band train as live. The 09:30 stamp replaces it. (7) The extension report still waits on Ido’s format.

**OPS DELTA 24 Sep 21:17 IDT — 3-hour briefing. Still no cluster reading since 14:45.**

- **Good.** The new sleeper is running (cycle 2 of 16, through about 04:14). The laptop still reaches the web: HTTPS to bgu.ac.il returned 200 on this wake. The last poll that reached Slurm, at 14:45, had all five jobs running and no tracebacks. Those jobs do not need this laptop. The scores at that poll stand: in-band probe **0.262** (ResNet-56 half **0.167**, tied with the episode-95 freeze, no new snapshot), 40-epoch probe **0.180** (ResNet-56 half **0.090**), area **0.054**, PPO-8 **0.060**, factored **0.036** with no freeze.
- **Bad.** `slurm.bgu.ac.il` port 22 has timed out on every wake since 15:14, including this one, twice. The blind spot is now about six hours. There is still no reading for the probes that follow 14:45 (in-band 252, 40-epoch 156, area 144, PPO-8 144, factored 108). A failed job in that window would not show up here.
- **Insight.** This is still the laptop path to the login node. Six hours of silence does not move the ranking. The best tested actor remains the in-band linear train, episode 95, **−7.1 @ 0.756** (§123). The 14:45 tie left the ResNet-56 half at **0.167**. The 40-epoch probe at episode 144 left that half at **0.090**.
- **Fable additions.** (1) Do not invent a score for the gap after 14:45. (2) Do not TEST from this gap. The tested points remain in-band episode 95 (§123) and 40-epoch episode 59 (§112). (3) Do not scancel anything from the 14:45 poll. (4) The first poll after the login returns is a catch-up. Quote a new probe only from a `PROBE` line. A tie is not a new freeze. (5) Do not compare the area-kind scores with the older structural probe near 0.26. (6) The 23:00 canvas, if the login is still down, is a stale stamp of the 14:45 scores. (7) The free GPU still needs a cell. Ops will not invent one. (8) The extension report still waits on Ido’s format.

**OPS DELTA 24 Sep 18:14 IDT — 3-hour briefing. No cluster reading since 14:45.**

- **Good.** The sleeper is running. The laptop still reaches the web: HTTPS to bgu.ac.il returned 200 on this wake. The last poll that reached Slurm, at 14:45, had all five jobs running and no tracebacks. Those jobs do not need this laptop. The scores at that poll stand: in-band probe **0.262** (ResNet-56 half **0.167**, tied with the episode-95 freeze, no new snapshot), 40-epoch probe **0.180** (ResNet-56 half **0.090**), area **0.054**, PPO-8 **0.060**, factored **0.036** with no freeze.
- **Bad.** `slurm.bgu.ac.il` port 22 has timed out on every wake since 15:14, including this one, twice. There is no reading for the probes that follow 14:45 (in-band 252, 40-epoch 156, area 144, PPO-8 144, factored 108). A failed job in that window would not show up here.
- **Insight.** This is the laptop path to the login node. Three hours of silence leaves the ranking where it was. The best tested actor is still the in-band linear train, episode 95, **−7.1 @ 0.756** (§123). Neon-raw is next (**−7.4 @ 0.757**). The mild-clone cluster sits near **0.92** keep. The 40-epoch fine-tune, the finished v3 and V4 trains, and layer-replacement as a training objective stay dead as a next train. The three live trains are still untested.
- **Fable additions.** (1) Do not invent a score for the gap after 14:45. (2) Do not TEST from this gap. The tested points remain in-band episode 95 (§123) and 40-epoch episode 59 (§112). (3) Do not scancel anything from the 14:45 poll. (4) The first poll after the login returns is a catch-up. Quote a new probe only from a `PROBE` line. A tie is not a new freeze. (5) Do not compare the area-kind scores with the older structural probe near 0.26. (6) The free GPU still needs a cell. Ops will not invent one. (7) The extension report still waits on Ido’s format.

**OPS DELTA 24 Sep 15:14 IDT — 3-hour briefing. In-band tied its freeze. ResNet-56 did not move.**

- **Good.** At 14:45 all five jobs were running, no tracebacks. The in-band probe at episode 240 is **0.262**, up from the **0.000** identity walk, and equal to the episode-95 freeze. The area critic rose from **0.73** to **0.92**. The factored critic is **0.91**, and the clip fraction fell from **0.56** to **0.21**. QOS was 5/6.
- **Bad.** That **0.262** did not write a new freeze. The ResNet-56 half is **0.167**, the same as episode 216. The rise is the ResNet-20 half, **0.358**. The 40-epoch probe at episode 144 is **0.180**, down from **0.224**, and its ResNet-56 half is back at **0.090**. It stays under the episode-59 freeze at **0.268**. Area’s probe is still **0.054**, under **0.059**. PPO-8’s probe is still **0.060**, under **0.063**, with critic **0.96**. Factored’s probe is still **0.036**, its best is still **0.049**, and there is still no freeze. This 15:14 wake could not reach the login. Port 22 timed out twice. There is no reading after 14:45.
- **Insight.** Matching the freeze number is not a return to the tested snapshot. The ResNet-56 half that carries the 0.756 point is the same **0.167** it was before the empty walk. The 40-epoch train gave that half back: **0.167** at episode 132, **0.090** at episode 144.
- **Fable additions.** (1) Do not TEST episode 240. A tie is not a new freeze. The tested in-band point remains episode 95 (§123). Do not scancel that train. (2) Do not TEST the 40-epoch episode 144. The tested point remains episode 59 (§112). (3) Do not TEST area episode 132, PPO-8 episode 128, or factored episode 96. (4) Do not scancel the factored train. Do not cross off that head from telemetry. (5) Do not compare these area-kind scores with the older structural probe near 0.26. (6) The free GPU still needs a cell. Ops will not invent one. (7) The extension report still waits on Ido’s format. (8) The next poll is a catch-up. Do not invent a score for the gap after 14:45.

**OPS DELTA 24 Sep 12:13 IDT — 3-hour briefing. The empty in-band walk is still the last probe.**

- **Good.** All five jobs are still running, no tracebacks. The in-band train is on episode 236, past the zero probe, and it has not stopped. The area probe is still **0.050**. The factored probe at episode 96 is **0.036**, flat against episode 84. QOS is 5/6.
- **Bad.** No new in-band probe. Episode 228 is still **0.000**, under the episode-95 freeze at **0.262**. The 40-epoch probe is still **0.224**. The PPO-8 probe at episode 128 is **0.060**, down from **0.061**, and the ResNet-56 half is **0.027**. It stays under the freeze at **0.063**. The area best is still **0.059**, and the critic fell from **0.80** to **0.73**. The factored best is still **0.049**, there is still no freeze, and update 25 had clip fraction **0.56**.
- **Insight.** Three hours after the identity walk, the live in-band policy has not been replaced and has not been re-scored. The new trains did not beat a freeze. PPO-8’s approach toward **0.063** reversed by a thousandth. A factored critic of **0.79** with a flat probe is still not a snapshot.
- **Fable additions.** (1) Do not TEST the zero probe, and do not scancel the in-band train. The tested point remains episode 95 (§123). The stop is blocked until episode 250. (2) Do not TEST PPO-8 episode 128, area episode 120, or factored episode 96. The tested 40-epoch point remains episode 59 (§112). (3) Do not scancel the factored train for the clip fraction. Do not cross off that head from telemetry. (4) Do not compare these area-kind scores with the older structural probe near 0.26. The in-band **0.000** is an empty structural walk. (5) The free GPU still needs a cell. Ops will not invent one. (6) The extension report still waits on Ido’s format.

**OPS DELTA 24 Sep 09:13 IDT — 3-hour briefing. In-band probe walked without pruning.**

- **Good.** All five jobs are still running, no tracebacks. The area probe at episode 120 is **0.050**, up from **0.047**. The PPO-8 critic is **0.93**, and its probe is still **0.061**. The factored train is still running. QOS is 5/6.
- **Bad.** The in-band probe at episode 228 is **0.000** on both nets. The walk kept every layer (`rate=1.0`, reward 0). It was **0.210** at episode 216. The freeze stays episode 95 at **0.262**. At that probe the patience counter was **132/150**, the run is now on episode 231, and it will not stop before episode 250. Three rewinds have already happened. The 40-epoch probe is still **0.224**, last logged at 08:05. Area’s **0.050** is under the episode-83 freeze at **0.059**, and the ResNet-56 half fell to **0.023**. PPO-8 did not beat **0.063**. Factored’s probe is still **0.036**, its best is still **0.049**, and there is still no freeze. Its critic is **0.75**.
- **Insight.** The train that produced the tested 0.756 point just scored zero on the probe by refusing to prune. That does not replace episode 95. The live policy is moving away from the freeze. Area’s combined score ticked up while its ResNet-56 half did not. A critic of **0.93** is still **0.002** under PPO-8’s own freeze.
- **Fable additions.** (1) Do not TEST the zero probe, and do not scancel the in-band train. The tested point remains episode 95 (§123). The stop is blocked until episode 250. (2) Do not TEST area episode 120 or episode 83. Do not TEST PPO-8 episode 112. The tested 40-epoch point remains episode 59 (§112). (3) Do not scancel the factored train. Do not cross off that head from telemetry. (4) The in-band **0.000** is an empty walk on the structural probe. Do not read it as an area-kind score, and do not compare the area-kind scores with the older probe near 0.26. (5) The free GPU still needs a cell. Ops will not invent one. (6) The extension report still waits on Ido’s format.

**OPS DELTA 24 Sep 06:13 IDT — 3-hour briefing. New-train probes edged up. Long trains did not.**

- **Good.** All five jobs are still running, no tracebacks. The PPO-8 probe at episode 112 is **0.061**, up from **0.054**, with the ResNet-56 half at **0.031**. Its critic is **0.91**. The factored probe at episode 84 is **0.036**, up from **0.030**, with the ResNet-56 half at **0.022**. The area critic held at **0.76**. QOS is 5/6.
- **Bad.** Neither long train printed a new probe. In-band is still **0.210**, under the episode-95 freeze at **0.262**. The 40-epoch probe is still **0.224**, under the episode-59 freeze at **0.268**. PPO-8’s **0.061** did not beat its freeze at **0.063**. The area probe is **0.047**, under the episode-83 freeze at **0.059**, and the best score is still **0.059**. The factored best is still **0.049**, there is still no freeze, and the critic fell from **0.93** to **0.74** on update 22.
- **Insight.** Three hours moved the two weaker area-kind probes up by less than a hundredth, and left the tested snapshots where they were. PPO-8 is now **0.002** under its own freeze and still did not replace it. A critic of **0.91** is still not a snapshot. The new trains’ ResNet-56 halves are **0.024**, **0.031**, and **0.022**, against **0.167** on both long trains.
- **Fable additions.** (1) Do not TEST any current snapshot. The tested in-band point remains episode 95 (§123). The tested 40-epoch point remains episode 59 (§112). (2) Do not TEST PPO-8 episode 112 or area episode 83. (3) Do not scancel the factored train for the critic dip. Do not cross off the factored head from telemetry. (4) Do not compare these area-kind scores with the older structural probe near 0.26. (5) The free GPU still needs a cell. Ops will not invent one. (6) The extension report still waits on Ido’s format.

**OPS DELTA 24 Sep 03:11 IDT — 3-hour briefing. Both long-train probes held. ResNet-56 halves match at 0.167.**

- **Good.** All five jobs are still running, no tracebacks. The in-band probe at episode 216 is **0.210**, the same number as episode 204, including both halves (ResNet-56 **0.167**, ResNet-20 **0.254**). The 40-epoch probe at episode 132 is **0.224**, flat against **0.224**, and its ResNet-56 half recovered from **0.090** to **0.167**. The area critic recovered from **0.44** to **0.76**, and the entropy coefficient is back at **0.008**. QOS is 5/6.
- **Bad.** **0.210** is still under the episode-95 freeze at **0.262**. The 40-epoch combined score did not move because the ResNet-20 half fell from **0.358** to **0.281**. It stays under the episode-59 freeze at **0.268**. The area probe is still **0.049**, under the episode-83 freeze at **0.059**. PPO-8’s probe is still **0.054**, under **0.063**, and its critic is **0.77**. The factored probe is still **0.030**, its critic is **0.66**, and it still has no freeze.
- **Insight.** The two long trains now print the same ResNet-56 half, **0.167**. That is a recovery from **0.090**, and it is still short of a freeze. The in-band score reproduced to the third decimal, which is stability, not a new snapshot. The 40-epoch combined score stayed put because ResNet-20 gave the gain back.
- **Fable additions.** (1) Do not TEST episode 216 or episode 132. The tested in-band point remains episode 95 (§123). The tested 40-epoch point remains episode 59 (§112). (2) Do not TEST area episode 83. (3) Do not scancel the factored train. (4) Do not compare these area-kind scores with the older structural probe near 0.26. (5) The free GPU still needs a cell. Ops will not invent one. (6) The extension report still waits on Ido’s format.

**OPS DELTA 24 Sep 00:11 IDT — 3-hour briefing. Login returned. No test in the gap.**

- **Good.** All five jobs are still running, no tracebacks. The in-band probe at episode 204 is **0.210**, up from **0.202**. Its ResNet-56 half recovered from **0.090** to **0.167**. The area critic recovered from **0.69** to **0.88**, and it froze episode 83 at **0.059**, above the old **0.055**. The factored critic recovered from **0.55** to **0.85**. QOS is 5/6.
- **Bad.** **0.210** is still under the episode-95 freeze at **0.262**. The area probe at episode 96 fell to **0.049**, under the new freeze, and its ResNet-56 half is **0.027**. PPO-8’s probe at episode 96 is **0.054**, flat, under its freeze at **0.063**. The factored probe at episode 72 fell to **0.030** (ResNet-56 **0.016**), and it still has no freeze. The 40-epoch probe is still **0.224**, with the ResNet-56 half still **0.090**.
- **Insight.** The six-hour blind spot did not hide a collapse or a win. The one new freeze is an area-kind score of **0.059**, and the next probe already went under it. ResNet-56 halves are now **0.167**, **0.090**, **0.027**, **0.028**, and **0.016**.
- **Fable additions.** (1) Do not TEST area episode 83. Do not TEST in-band episode 204. The tested in-band point remains episode 95 (§123). The tested 40-epoch point remains episode 59 (§112). (2) Do not scancel the factored train. (3) Do not compare these area-kind scores with the older structural probe near 0.26. (4) The free GPU still needs a cell. Ops will not invent one. (5) The extension report still waits on Ido’s format.

**OPS DELTA 23 Sep 21:11 IDT — 3-hour briefing. No cluster reading since 18:09.**

- **Good.** The sleeper is running. The last poll that reached Slurm, at 18:09, had all five jobs running and no tracebacks. Those jobs do not need this laptop. The scores at that poll stand: in-band probe **0.202** (ResNet-56 half **0.090**), 40-epoch probe **0.224** (ResNet-56 half **0.090**), PPO-8 probe **0.053**, factored probe **0.049** with no freeze, area probe **0.024**.
- **Bad.** `slurm.bgu.ac.il` port 22 has timed out on every wake since 18:47. There is no reading for probes that were due after 18:09 (area 84, PPO-8 96, factored 72, in-band 204, 40-epoch 132). A failed job in that window would not show up here.
- **Insight.** This is the laptop path to the login node, not a result from the trains. Do not invent a score for the gap, and do not treat the 18:09 probes as newer than they are.
- **Fable additions.** (1) Do not TEST from this gap. The tested points remain in-band episode 95 (§123) and 40-epoch episode 59 (§112). (2) Do not scancel anything from a stale poll. (3) The first poll after the login returns is a catch-up. (4) The free GPU still needs a cell. Ops will not invent one. (5) The extension report still waits on Ido’s format.

**OPS DELTA 23 Sep 18:09 IDT — 3-hour briefing. ResNet-56 halves did not move.**

- **Good.** All five jobs are still running, no tracebacks. The login node answered again after the 15:39 timeout. PPO-8 cleared the update-10 check: critic **0.89**, and the last eight gaps average **+0.14**. The factored probe at episode 60 rose to **0.049** from **0.040**. The in-band train finished a 69-minute MobileNet episode and is on episode 199. QOS is 5/6.
- **Bad.** The 40-epoch probe at episode 120 is **0.224**, flat against **0.225**, and the ResNet-56 half is still **0.090**. It stays under the episode-59 freeze at **0.268**. The in-band probe is still **0.202**, with the same ResNet-56 half of **0.090**, under the episode-95 freeze at **0.262**. PPO-8’s probe at episode 80 is **0.053**, and its ResNet-56 half fell to **0.020**. Factored’s **0.049** is the ResNet-20 half (**0.079**); the ResNet-56 half is **0.020**, and there is still no freeze. Its critic was **0.55** at update 15. The area critic fell from **0.90** to **0.69** at update 20, with clip fraction **0.43**. Its best is still **0.055**, and its probe is still **0.024**.
- **Insight.** Three hours of probes left every ResNet-56 half where it was. In-band and the 40-epoch train both read **0.090** on that half. The three new trains read **0.019**, **0.020**, and **0.020**. The only combined score that rose was factored, and the rise is ResNet-20. Clearing the gap bar, as PPO-8 did, is not a snapshot.
- **Fable additions.** (1) Do not TEST episode 192, episode 120, episode 80, or factored 0.049. The tested in-band point remains episode 95 (§123). The tested 40-epoch point remains episode 59 (§112). (2) Do not scancel the factored train or the area train for the critic dips. (3) PPO-8 passing the update-10 gap check is not a test. (4) Do not compare these area-kind scores with the older structural probe near 0.26. (5) The free GPU still needs a cell. Ops will not invent one. (6) The extension report still waits on Ido’s format.

**OPS DELTA 23 Sep 15:09 IDT — 3-hour briefing. In-band rose on ResNet-20. ResNet-56 fell.**

- **Good.** All five jobs are still running, no tracebacks. The in-band probe at episode 192 is **0.202**, up from **0.168** at episode 180. The factored critic recovered from **0.72** at update 13 to **0.89** at update 14. The area probe at episode 72 is **0.024**, up from **0.021**. QOS is 5/6.
- **Bad.** The in-band rise is the ResNet-20 half, **0.314** (was **0.139**). The ResNet-56 half fell from **0.196** to **0.090**. **0.202** is still under the episode-95 freeze at **0.262**, and there is no new freeze. The 40-epoch probe is still **0.225**, and episode 118 on skinny ResNet-56 had a gap of **+0.007**. The area best is still **0.055**. PPO-8’s probe is still **0.053**, under its freeze at **0.063**. Factored’s best is still **0.040**, the ResNet-56 half is **0.017**, and it still has no freeze.
- **Insight.** The combined in-band score moved because the two halves moved in opposite directions. ResNet-56 on that probe is back at the same **0.090** the 40-epoch probe has been showing. The tested point remains episode 95. The three new trains’ ResNet-56 halves are **0.019**, **0.028**, and **0.017**. A recovered critic is still not a snapshot.
- **Fable additions.** (1) Do not TEST episode 192. The tested in-band point remains episode 95 (§123). The tested 40-epoch point remains episode 59 (§112). (2) Do not TEST area 0.024 or factored 0.040. (3) Do not scancel the factored train. (4) Do not compare these area-kind scores with the older structural probe near 0.26. (5) The free GPU still needs a cell. Ops will not invent one. (6) The extension report still waits on Ido’s format.

**OPS DELTA 23 Sep 12:06 IDT — 3-hour briefing. New-train ResNet-56 halves stayed near 0.02.**

- **Good.** All five jobs are still running, no tracebacks. The 40-epoch train is on episode 115, fine-tuning MobileNet (epoch 20/40). The factored train is on episode 51, also fine-tuning MobileNet. Its probe at episode 48 rose to **0.040** from **0.026**. QOS is 5/6.
- **Bad.** That 0.040 is the ResNet-20 half (**0.063**). The ResNet-56 half is **0.017**, and there is still no freeze. The area probe at episode 60 is **0.021** (ResNet-56 **0.016**), under the freeze at **0.055**. The area critic slipped from **0.93** at update 16 to **0.86** at update 17; the best score is still **0.055**. PPO-8’s probe at episode 64 is **0.053**, under its freeze at **0.063**, and its critic is **0.82**. The in-band probe is still **0.168**, under episode 95. The 40-epoch probe is still **0.225**, under episode 59.
- **Insight.** In three hours the only selection score that rose was the factored probe, and the rise is not on ResNet-56. The three new trains’ ResNet-56 halves are **0.016**, **0.028**, and **0.017**. A healthy critic is still not a snapshot worth testing.
- **Fable additions.** (1) Do not TEST any current snapshot. The tested in-band point remains episode 95 (§123). The tested 40-epoch point remains episode 59 (§112). (2) Do not treat factored 0.040 as a freeze, and do not scancel it. (3) Do not compare these area-kind scores with the older structural probe near 0.26. (4) The free GPU still needs a cell. Ops will not invent one. (5) The extension report still waits on Ido’s format.

**OPS DELTA 23 Sep 09:06 IDT — 3-hour briefing. Scores did not move.**

- **Good.** All five jobs are still running, no tracebacks, about 22 h on the three new trains. The in-band probe at episode 180 is **0.168**, up from **0.153** at episode 168. The ResNet-56 half went from 0.167 to **0.196**. The area critic is **0.89** at update 15. The factored batch score reached **0.49** at update 11, with critic **0.90**. QOS is 5/6.
- **Bad.** **0.168** is still under the episode-95 freeze at **0.262**. The ResNet-20 half of that probe is stuck at **0.139**. The 40-epoch probe is still **0.225**. No new snapshot: area’s best is still **0.055** (episode 24), PPO-8’s best is still **0.063** (episode 32, and episode 48 was 0.056), factored’s best is still **0.026** with no freeze. Episode 59 on the area train has a gap of **+0.45**; that is one episode, not a probe.
- **Insight.** Three more hours of training did not improve a selection score. The critics stay high while the scores that decide a snapshot stay where they were at dawn. A single large gap does not change that.
- **Fable additions.** (1) Do not TEST any current snapshot. The tested in-band point remains episode 95 (§123). The tested 40-epoch point remains episode 59 (§112). (2) Do not scancel the factored train. (3) The free GPU still needs a cell. Ops will not invent one. (4) The extension report still waits on Ido’s format.

**OPS DELTA 23 Sep 06:06 IDT — 3-hour briefing. Both old probes fell. Factored missed the gap bar.**

- **Good.** All five jobs are still running, no tracebacks. The area train is on PPO update 13 with critic `ev=0.917` and `batch_score=0.406`. PPO-8’s critic is `ev=0.932`. The factored critic is `ev=0.742` at update 10, up from the 0.15 dip at update 3. QOS is 5/6. The 04:06 loop is the sleeper.
- **Bad.** ft40 probe 108 is **0.2249**, down from 0.2622 at probe 96. The ResNet-56 half is **0.090**; ResNet-20 held **0.360**. Freeze stays ep0059 / 0.2679. Do not auto-TEST. Area probe 48 is **0.026**, under its freeze at 0.055. PPO-8 probe 48 is **0.056**, under its freeze at 0.063. Factored’s last eight gaps average **+0.017** at update 10, under the +0.05 line. Its probe 36 is **0.026** and it still has no snapshot. In-band’s latest probe is still **0.1532**; episode 180 has not scored yet.
- **Insight.** The two long trains are moving away from the snapshots we already tested. The three new trains are not replacing them: their area scores are flat-to-down around 0.03–0.06. Factored’s critic is healthy and its actions are still near uniform, which is the same split the area train showed earlier in the other direction (uniform-leaving actions, weak score). A telemetry miss is not the cross-off. The factored head is crossed off only if a later TEST matches the area baseline at equal keep.
- **Fable additions.** (1) Do not TEST ft40 ep0059 again, and do not TEST the new snaps (area ep0023, PPO-8 ep0031). (2) Factored failed the update-10 gap check. Do not scancel it. Do not cross off the factored head from this telemetry. (3) PPO-8 is no longer climbing; probe 48 is below probe 32. (4) The free GPU still needs a cell. Ops will not invent one. (5) The extension report still waits on Ido’s format.

**OPS DELTA 23 Sep 03:06 IDT — 3-hour briefing, after the laptop slept through the flight.**

- **Good.** All five jobs are still running, no tracebacks. Area `21536396` reached PPO update **11**: critic `ev=0.937`, and the last eight episodes average gap **+0.18** (every one above +0.11). That clears the old update-10 telemetry bar. PPO-8 `21536397` is on update 5, `ev=0.883`, last eight gaps average **+0.07**, and it froze a second snap: ep0031 area-score **0.0625** (first freeze was ep0015 / 0.0568). The factored critic recovered from 0.15 at update 3 to **0.784** at update 8. ft40 probe 96 is **0.2622**, the same ceiling as probe 84. The 09:15 loop is still the sleeper; the laptop sleep stretched the 30-minute timer.
- **Bad.** In-band probes fell: ep156 **0.2104**, ep168 **0.1532** (r20-w10 half 0.139, was 0.360 at ep144). Freeze stays ep0095 / 0.2622. Do not auto-TEST, and do not read episode 173’s gap of +0.245 as a probe. Area-kind scores are small: area ep36 is **0.020** after a freeze at ep0023 / **0.055**; factored ep24 is **0.022** and has no freeze. Factored’s last eight gaps average **+0.020**, under the +0.05 line, at update 8. QOS is still **5/6**. The empty GPU stayed empty overnight.
- **Insight.** A non-uniform policy and a useful area score are different facts. Area’s actions left uniform (gap +0.18, critic 0.94) while its area-kind probe is 0.02–0.06. PPO-8 is the only new train whose later probe beat its first freeze, and 0.0625 is still an early score, not a TEST. In-band’s falling probe says the live linear train is moving away from the ep0095 snapshot that already matched the 0.756 point. Factored’s critic came back; its actions did not.
- **Fable additions.** (1) Do not TEST ep0023, ep0015, or ep0031. Area-kind 0.06 is not a go. (2) Area passes the update-10 gap-and-critic check. Factored does not, yet; it has two updates left before that bar. Do not scancel any of them. (3) In-band’s live probe is below its freeze. Leave ep0095 as the tested snapshot (§123). (4) ft40 probe 96 did not beat ep0059. (5) The free GPU still needs a cell. Ops will not invent one while Ido is away. (6) Do not draft the extension report until the format arrives.

**OPS DELTA 22 Sep 15:56 IDT — Ido closing the PC for the airport. 3-day trip; PC stays on at the hotel.**

- **Good.** Day’s TESTs stand. Learned-schedule: 3-pass mild §114 r56 **−6.9 @ 0.923**, 3-pass L1 §122 **−7.6 @ 0.914**. Neither reaches ~0.75 in band. ep0095 §123 r56 **−7.1 @ 0.756**, `state_used` **38% / 53%**. Catalog L mild §124 r56 **−3.3 @ 0.661** (same keep as §103). Three new trains R at FT 12, no tracebacks, ~5 h: area update 3 `ev=0.815`; PPO-8 update 1 `ev=0.751`; factored update 3 `ev=0.150` (down from 0.52). Slurm keeps running with the laptop shut.
- **Bad.** VGG-19 C100 val_best is the unpruned net on both heuristics (§124/§125). Terminals are val about −31 to −34 pp; do not quote the pass 2/2 summary. No LR passed both the C100 gate and the C10 thin control. Probes unchanged: in-band **0.2249**, ft40 **0.2622**. QOS **5/6** since 11:24. The empty GPU has no legal job. New-train gaps are still early (area ep12 **+0.064**, PPO-8 ep13 **+0.014**, factored ep11 **+0.029**).
- **Insight.** The 0.756 r56 point is a schedule the 3-pass heuristics do not select inside the band, and a second snapshot walks to it. The encoder is read. Catalog L cuts CIFAR-10 inside the band and leaves CIFAR-100 VGG-19 on the first cut. A high critic is not a non-uniform policy; the update-10 gap check has not arrived. One FT recipe is still open, and the three trains have already started at 12 epochs.
- **Fable additions.** (1) Write the learned-schedule sentence from §114/§122; the actor point is §123. (2) Do not park representation. (3) VGG-19 C100 under recipe A 2-pass has no in-band cut. (4) Do not emit the 16-net catalog. Do not resubmit `21536396/97/98`. (5) Do not call the three trains before update ~10. Watch the factored critic (0.52 → 0.15 at update 3); do not kill it. (6) The free GPU needs a cell from the sitting. A-LSQ, C-PCA, and cost-denominated STOP are unimplemented. Ops will not invent the filler. (7) Do not auto-TEST. (8) Do not draft the extension progress report until Ido sends the format.

**OPS DELTA 22 Sep 15:15 IDT — 3-hour briefing. Trains only; no new TEST.**

- **Good.** The three new trains are alive at FT 12 with no tracebacks. Area `21536396` (~4 h) PPO update 3: critic `ev=0.815`, `batch_score=0.338`. PPO-8 `21536397` update 1: `ev=0.751`, episode 12. Factored `21536398` update 2: `ev=0.516`; episodes 8 and 10 have gap **+0.057** and **+0.054**. In-band episode 150 (chenyaofo ResNet-56) gap **+0.099**. The 40-epoch train finished the long MobileNet episode and is on episode 92 (VGG-13 gap **+0.113**).
- **Bad.** No new probe. In-band is still **0.2249** (episode 144). ft40 is still **0.2622** (episode 84). Neither freeze moves, so nothing to TEST. Area’s latest episode gap is **+0.030**, under the +0.05 line, and that train is on update 3 of the update-10 check. PPO-8’s latest gap is **+0.024**. QOS is still **5/6**. The free GPU has been empty since the Catalog L twins finished at 11:24.
- **Insight.** A high critic score is not a non-uniform policy. Area’s explained variance is already 0.82 while the action gap on the latest episode is near zero. The go/no-go is the mean gap at update ~10, and none of the three trains is there. The large gaps on in-band episode 150 and ft40 episode 92 are single training episodes, not probe scores.
- **Fable additions.** (1) Do not call the three trains yet. Wait for update ~10. (2) The free GPU still needs a cell from the sitting. A-LSQ, C-PCA, and cost-denominated STOP are unimplemented. Ops will not invent the filler. (3) Do not auto-TEST ep0095 or ep0059. (4) The 12:15 points stand: learned-schedule from §114/§122, representation stays (`state_used` 38%/53%), VGG-19 C100 has no in-band cut, do not emit the 16-net catalog, do not resubmit the three trains.

**OPS DELTA 22 Sep 12:15 IDT — 3-hour briefing. V7 eval list is finished.**

- **Good.** Learned-schedule stands. 3-pass L1 (**§122**, `21536387`, 12 h 59 m, exit 0) selects r56 **−7.6 @ 0.914/0.748**, val −9.94. 3-pass mild (§114) was −6.9 @ 0.923. Neither reaches ~0.75 inside the band. ep0095 (**§123**, `21536395`, exit 0) reproduces ep0083: r56 **−7.1 @ 0.756/0.691**, val −9.68. `state_used` is **38.1%** on r20 (42 steps) and **52.6%** on r56 (114 steps). Catalog L mild (**§124**) r56 is **−3.3 @ 0.661**, val −8.18, the same keep as §103 and 0.6 pp kinder. VGG-16 mild is **−3.5 @ 0.657**. L1 (**§125**) reaches that same −3.5 pp on VGG-16 at **0.411** kept. Area `21536396`, PPO-8 `21536397`, and factored `21536398` are all R at train FT 12, episode 2, no tracebacks.
- **Bad.** VGG-19 CIFAR-100 selects the **unpruned** net on both heuristics (val_best 0.0 @ 1.000). Floor and terminal sit near val **−30 to −34 pp**. The log’s pass 2/2 summary (mild 0.657, L1 0.412) is that terminal. Do not quote it. No learning rate admitted CIFAR-100 and held the CIFAR-10 control, so the 16-net catalog stays off. In-band probe 144 = **0.2249**, below the ep0095 freeze. ft40 is on episode 89; probe 84 is still **0.2622**. QOS is **5/6**. The V7 submit list is empty.
- **Insight.** The 0.756 r56 point is a schedule the 3-pass heuristics do not select inside the band, and a second in-band snapshot walks to the same point. The encoder changes the action on both thin nets, so representation is not decoration. Catalog L’s CIFAR-10 twins cut in band; the CIFAR-100 VGG leaves the band on the first real cut. One fine-tune recipe is still the open decision, and the three new trains have already started at 12 epochs, so that decision does not rewrite them.
- **Fable additions.** (1) Write the learned-schedule sentence from §114 and §122; ep0095 §123 is the matching actor point. (2) Do not park the representation cell: `state_used` 38% / 53%. Do not start a new representation train from ops. (3) Catalog L bar-2 is in. VGG-19 C100 under recipe A 2-pass has no in-band cut. (4) One FT recipe is still the open question. Do not emit `database_offline_v7_diverse_admitted.json`. Do not resubmit `21536396/97/98`. (5) The free GPU needs a cell from this sitting. A-LSQ, C-PCA, and cost-denominated STOP are still unimplemented. Ops will not invent the filler. (6) Do not auto-TEST in-band ep0095 or ft40 ep0059 again.

**OPS DELTA 22 Sep 09:10 IDT — catch-up after a 04:01–09:04 monitoring gap. 3-hour briefing included.**

- **Good.** Both CIFAR-100 re-gates finished cleanly. Adam 1e-4 (**§117**) admits **4/8**: VGG-11 −4.8 @ 0.722, VGG-13 −4.9 @ 0.805, MobileNet-v2×1 −2.4 @ 0.801, DenseNet-40 −4.1 @ 0.944. SGD 0.01 (**§121**) admits **2/8**, the two VGGs, at a deeper keep (0.66). The 12/4 Adam 1e-3 thin reference (**§120**) is −5.3 @ 0.536 and −6.5 @ 0.933. Catalog L mild r56, still partial, is **−3.3 @ 0.661** val −8.18, the same keep as §103 (−3.9 @ 0.661). The ep0095 counterfactual on r20 uses state on **38%** of steps. QOS stayed full the whole gap. Slurm did not idle.
- **Bad.** Both gentler learning rates **fail** the CIFAR-10 thin control. Adam 1e-4 (§118): r20 −8.3 @ 0.655 (3.0 pp worse and shallower than §120), r56 −8.0 @ 0.930 (1.5 pp worse, same keep). SGD (§119): r20 −8.1 @ 0.560, r56 −7.8 @ 0.930 (1.3 pp worse). The 4/8 Adam count does **not** unlock the 16-net catalog. ResNets admit on neither arm. In-band probe 132 = **0.2235**, below the ep0095 freeze. ft40 probe 84 = **0.2622**, back at the old ceiling. 3-pass L1 is 11 h in, r56 step ~146, still no `val_best`. The ops `[cf]` grep anchored at line start misses the log prefix; the lines are there.
- **Insight.** A gentler fine-tune recovers some CIFAR-100 families and costs CIFAR-10. There is no single learning rate in this pair that admits CIFAR-100 and also holds the CIFAR-10 control. The written rule is one recipe for every net. Per-dataset learning rates are a Fable decision, not an ops resubmit.
- **Fable additions.** (1) Do not emit `database_offline_v7_diverse_admitted.json`. Leave `21536396/97/98` at 12/4 on p5b2. (2) Decide whether one FT recipe is still mandatory once Adam 1e-4 admits VGG + MobileNet×1 + DenseNet and fails thin C10. (3) Learned-schedule still waits on 3-pass L1 r56; 3-pass mild §114 did not reach 0.75 in band. (4) ep0095 r20 `state_used` 38% means the encoder is not decoration on that net; wait for the r56 `val_best` before a representation GO. (5) Catalog L mild r56 reproduced the §103 keep; VGG-16 and VGG-19 are still walking. (6) Monitoring died at 04:01 because the one-shot sleeper exited and the agent never took the turn. Cluster work was not lost.

**OPS DELTA 22 Sep 02:58 IDT — 3-hour briefing.**

- **Good.** Adam 1e-4 re-gate is **2 admits / 5 finished**: VGG-11 **−4.8 @ 0.722/0.694** val −9.83; VGG-13 **−4.9 @ 0.805/0.768** val −8.38. Both inside τ at a real cut. SGD arm **`21536389` R** (~13 min) with `optim=sgd lr=0.01` confirmed, on r20-w13. QOS stayed **6/6**. v3-fpgm **`21385158` COMPLETED** 02:45 exit 0 (§115); its GPU is the SGD job. In-band reached ep128, gap **+0.185**.
- **Bad.** The three CIFAR-100 ResNets are not admits (kept 0.986 / 0.999 / 0.984). 2/5 is short of the 4/8 catalog unlock, and three nets remain. 3-pass L1 **`21536387`** is 5 h in on r56 step 47, still no `val_best`. fpgm’s last probe was **0.210**, below the ep0011 freeze already TESTed as a mild clone (§95). ft40 train is alive (log heartbeat 02:59) but the last finished episode is still ep78. No 2 pp win.
- **Insight.** At Adam 1e-4 the gate splits by family: VGG selects a cut inside the band; ResNet stays near identity. A single learning rate does not unlock “CIFAR-100” as a block. If the last three nets (MobileNet ×0.5, now walking, plus ×1.0 and DenseNet-40) add fewer than two admits, the fallback is V7-lite (1–3 admits), not the 16-net catalog.
- **Fable additions.** Do not emit `database_offline_v7_diverse_admitted.json` until this arm finishes and the thin controls pass. Do not resubmit `21536396` overnight. Learned-schedule still waits on 3-pass L1 r56 (§114 mild did not reach 0.75 in band). SGD is the paired arm; quote it only when its eight `val_best` rows exist. fpgm snap stays §95.

**OPS DELTA 22 Sep 00:52 IDT — 3-pass mild catalog COMPLETED §114.** Job **`21536384`** exit 0, 4 h 2 m. r56 **−6.9 @ 0.923/0.769**, val **−9.49**, step 38. Same keep as 2-pass mild §93 (−6.6 @ 0.923). Floor-hold 0.702 is outside τ (val −14.27). This heuristic does **not** reach ~0.75 kept in band, so it does not erase the §111 gap. r20 **−7.4 @ 0.417/0.608**, val −7.75. 3-pass L1 **`21536387`** still on r56 (~step 19). Learned-schedule sentence waits on that row. Re-gate **`21536388` R** (~25 min), `optim=adam lr=0.0001` confirmed. First C100 net r20-w13: **−2.9 @ 0.986**, val −7.96 — kept above 0.98, not an admit. QOS 6/6. Do not scancel. Do not resubmit the area trains.

**OPS DELTA 21 Sep 23:47 IDT — 3-hour briefing (missed 20:46; Ido reconnected on Grok 4.7, then bed).** Canonical ops remains `docs/PROMPT_OPS_V7_QUEUE.md`.

- **Good.** §112 is in. ft40 TRAJ **`21535193` COMPLETED** (3 h 1 m, exit 0, `cs-pheno-05`). r56 **−7.1 @ 0.923/0.769**, val **−9.35**, step 38. r20 **−4.2 @ 0.536/0.655**, val **−5.65** (same 2-pass keep; not the comparison). The 40/10 resubmit rule **did not fire** (needs a deeper in-band keep than 0.923, or ≥ 1 pp kinder at equal keep). `21536396/97/98` stay PD at 12/4. QOS stayed **6/6**: TRAJ’s GPU is 3-pass mild **`21536384`** (R 3 h 21 m, r56 step ~142); svd’s GPU is 3-pass L1 **`21536387`** (R 1 h 48 m, r56 just started). Both logs show `optim=adam lr=0.001`.
- **Bad.** Probe **0.2679** selected the mild point: r56 is **0.5 pp worse** than §93 (−6.6 @ 0.923) and **0.3 pp worse** than §95 (−6.8 @ 0.923). The deeper cuts are outside τ (floor-hold step 91: val −15.06 @ 0.702). ft40’s next probe **72 = 0.2622**, back at the old ceiling. In-band probe **120 = 0.2514**, below freeze ep0095; ep122 gap **+0.006**. v3-svd **`21385159` COMPLETED** 21:59, ep249, last probe 0.2406, only snap still ep0011 (§97 / ledger §113). No 2 pp win.
- **Insight.** Train-FT length did not move the selected operating point. The only ranking-menu actor that selected a deeper *in-band* r56 keep is still in-band-linear §111 (0.756, 12/4). 3-pass r20 is already deep inside τ (mild **−7.4 @ 0.417**, L1 **−9.3 @ 0.319**). That does not answer the learned-schedule sentence; r56 of those two jobs does. r20 stays off the policy comparison.
- **Fable additions (next paste, not a new sitting from ops).** (1) Lock the 40/10 cell as closed for the resubmit: trains stay 12/4 `area`. (2) Wait for 3-pass r56 before writing the learned-schedule sentence. (3) The 0.2679 probe selecting 0.923 is evidence for the area selection score, which is queued as `21536396` and has not started. (4) A-LSQ, C-PCA, and cost-denominated STOP stay unimplemented; ops will not start them. (5) Partial controls, not ledgered until r56: 3-pass mild r20 −7.4 @ 0.417/0.608 val −7.75; 3-pass L1 r20 −9.3 @ 0.319/0.569 val −9.18. No overnight decision. Ido reviews diffs in the morning, then Gilad, then the university extension letter.

**OPS DELTA 21 Sep 19:25 IDT — Fable V7 queue is live; this file is no longer the submit lock.** Canonical overnight ops: `docs/PROMPT_OPS_V7_QUEUE.md` (supersedes V6 lock §2–§3). Thirteen PD jobs from `tree_v7`; leap `src/` untouched. First free GPU → `21536384` 3-pass mild. Ledger from **§112** (ft40 TRAJ, still walking r56 step 68). Two rewrite rules only while trains PD: (1) §112 40/10 walks differently → Ido/Fable confirms, then scancel/resubmit `21536396/97/98` with 40/10; (2) re-gate ≥4 C100 + thin-control pass → prepare `database_offline_v7_diverse_admitted.json`, Ido names it. Do not auto-TEST freezes. Do not scancel R. QOS **6/6**. PC closing for drive home — QOS PD auto-starts.

**OPS DELTA 21 Sep 17:46 IDT — 3-hour briefing. Hole filled: ft40 TRAJ `21535193` R.** QOS **6/6**. Job `traj-v5-ft40-ep0059` on `cs-pheno-05` (~23 min). Scratch `tree`, snap `job21443408/snapshots/ep0059`, profile `eval_c10_thin_traj`, det=1, traj=1, skip_train=1, 2-pass, group_once, `ft_recipe=A`, 5-action fpgm. **No val_best yet.** Do not submit a 7th. Do not overlay leap. In-band ep112, freeze still ep0095/0.2622, next probe 120. Paste-ready sitting prompt already given in ops chat 17:20; if Fable already ran, this is a follow-up: the rank-1 GPU is in flight.

**OPS DELTA 21 Sep 17:18 IDT — Ido asked for a paste-ready sitting prompt (this chat).** Cluster: QOS **5/6**, hole empty. In-band **21459737** ep111, PPO-28 `ev=0.727`, freeze **ep0095/0.2622**, probe **108 = 0.2618** (no new freeze), next **120**. ft40 **21443408** ep67, freeze **ep0059/0.2679**, next probe **72**. No new TEST since §111. Do not overlay leap. Do not auto-TEST. Do not fill the hole until this sitting locks it. Paste the in-chat prompt Ido is about to copy; this file is the long ops log + §7.

**OPS DELTA 21 Sep 14:39 IDT — 3-hour briefing (Ido sitting; do not nag).** QOS **5/6**, hole still empty. No new TEST since §111. In-band **ep107**, PPO-27 `ev=0.837` `batch=0.307` `best=0.262`, gap **+0.018**, freeze still **ep0095/0.2622**, probe **108 pending**. ft40 **ep64**, freeze still **ep0059/0.2679**. Lock unchanged: one GPU = ft40 TRAJ xor representation; not C-G DRL; not Catalog L DRL of this actor. Do not auto-TEST. Do not overlay leap. If this paste already went out, treat this as a follow-up delta only.

**OPS DELTA 21 Sep 11:31 IDT — 3-hour briefing.** QOS **5/6**, one hole empty on purpose. Leap `c08d513`. No new TEST since §111. Do not overlay leap. Do not auto-TEST. Ido pastes this file.

- **Good.** TRAJ **`21512868` COMPLETED §111**: r20 **−3.5 @ 0.536** (cloned mild keep); r56 **−7.1 @ 0.756** val −9.65 (≡ neonraw keep, **not** mild 0.923). Linear-in-band is the first *cbrt_cubes / ranking-menu* actor whose r56 keep left 90%. ft40 **`21443408` froze ep0059 / 0.2679** at probe 60 (10:24) — r56-w6 **0.178** / r20-w10 **0.358**. That **beats** first freeze 0.241 **and** the 0.262 cluster ceiling every 12-ep arm froze at. Isolated 12/4 vs 40/10 is now identifiable. In-band train still R ep99, PPO-25 `ev=0.543` `batch=0.379` `best=0.262`, freeze still ep0095/0.2622 (same ceiling as ep0083; do **not** auto-TEST).
- **Bad.** No Gilad 2 pp win. r20 still cloned keep. In-band gap on ep99 is **+0.016** (flattening toward uniform). fpgm PPO-55 `ev=−0.041` (spent control). svd/fpgm/bnscale still stuck at first-freeze 0.262 TESTed as mild. QOS hole is **not** C-G DRL.
- **Insight.** Two competing next-GPU cells, Fable must pick **one**: (A) skip-train TRAJ of **ft40 ep0059** (cheapest identification: longer train FT actually moved the probe); (B) representation overlay / §7.8 2b (thesis novelty; still no overlay written). Catalog L DRL of in-band is **not** clean 2a (actor is not a SOTA win) and **not** clean 2b (r56 left 90%). Ops will **not** fill the hole until this sitting locks it. Do not scancel remaining v3.

**OPS DELTA 21 Sep 08:54 IDT — in-band TRAJ COMPLETED §111. Mixed: r20 cloned keep; r56 did not.** Job **`21512868` COMPLETED** 08:24 exit 0, 3 h 2 m, `cs-pheno-08`. Quote TRAJ val_best: r20 **−3.5 @ 0.536/0.655** val −4.50; r56 **−7.1 @ 0.756/0.691** val −9.65 (step 58, inside τ). r20 ≡ mild keep. r56 keep **0.756** ≡ neonraw §99 **0.757**, **not** mild/V4 **0.923**. “First freeze ≡ mild keep” is **falsified on r56**. Not a 2 pp win. QOS **5/6** (one hole). **Do not fill with C-G DRL.** §7.8 is mixed: not clean 2a (actor is not a SOTA win) and not clean 2b (r56 left 90%). **Lock 2a vs 2b vs a §7.9 replacement in this sitting.** Ido can paste now. Do not overlay leap.

**OPS DELTA 21 Sep 08:22 IDT — 3-hour briefing.** QOS **6/6** (`MaxTRESPU=6`). TRAJ **`21512868` still R** (~3 h, `cs-pheno-08`). r20 **PRELIM −3.5 @ 0.536/0.655 val −4.50** ≡ mild keep §93. *Superseded: catalog COMPLETED §111.*

**OPS DELTA 21 Sep 06:18 IDT — in-band TRAJ r20 PRELIM; r56 still walking.** Job **`21512868`** `[eval] TRAJ val_best` skinny r20-w2: **acc 0.648 → 0.613 (−3.5 pp) | params x0.536 | FLOPs x0.655 | val −4.50 pp**. Equal keep vs 2-pass mild §93 **−3.4 @ 0.536**; 0.2 pp kinder than V4 §102 **−3.7 @ 0.536**. **Keep cloned 90% on r20.** Not a Gilad 2 pp win. Catalog **not** complete — job still R on r56-w4 (~step 19, identity then 0.9). Do **not** lock “first non-mild actor” from r20 alone (neonraw r56 was the net that did *not* clone keep). Ops will not ledger until r56 val_best. Do not overlay leap. Do not start C-G DRL.

**OPS DELTA 21 Sep 05:40 IDT — do not scancel v3 for C-G DRL.** Ido asked whether two low-yield R jobs should die so C-G / C-G+ *training* can start while Fable waits. Grok: **no.** Lowest-yield R are v3-svd `21385159` and v3-fpgm `21385158` (freeze stuck ep0011, rewind 3/3, already TESTed ≡ mild). That does **not** make C-G DRL the fill. Starting those trains now pre-empts §7.9. Keep TRAJ `21512868`, in-band `21459737`, ft40 `21443408`, bnscale `21385160` (untested later snap ep0107). Fable may override when Ido pastes.

**OPS DELTA 21 Sep 05:28 IDT — wait vs invent-actor; C-G-for-DRL; next directions.** Ido asked whether to paste Fable *now* to invent the first non-mild actor, or wait ~7 h. Grok: **wait for `21512868` TRAJ val_best** before any overlay / new train. Enough material exists to *design* and to *ponder* (do that in **§7.9** — Fable may kill Grok’s ranking). Not enough *outcome* to submit a new actor. TRAJ is 7 min in on skinny r20 (identity on illegal rows, 0.9 on legal — looks like the mild shape; **not a verdict**). Do not overlay leap.

**OPS DELTA 21 Sep 05:22 IDT — hole filled. QOS is 6 not 7. Lock §7.8 for the *next* GPU.** Ido asked to fill empty slots. The only secure method cell was the in-band freeze TRAJ: **`21512868` R** `traj-v6-inband-ep0083` on `cs-pheno-08` (rtx_3090), submitted from `tree_v6_inband`, not leap. `[policy_config]` pinned 2-pass, group_once, 5-action fpgm, `ft_recipe=A`, `align=next`, det=1, traj=1. Live `MaxTRESPU gres/gpu=6` (sacctmgr 05:22) — the “two holes” were a stale cap. QOS **6/6 full**. Do not enqueue a 7th. **Lock §7.8** (what runs when a GPU frees). Do not overlay leap.

**OPS DELTA 21 Sep 05:17 IDT — 3-hour briefing + TEST lock (Ido reconnect).** Ops will append a briefing here every ~3 h (good / bad / insight) so this paste stays current. Experiment-plan canvas is dropped; do not restamp it. Do not overlay leap.

**OPS DELTA 21 Sep 05:15 IDT — in-band froze. Ping Ido. Do not auto-TEST.** Job **`21459737` Snapshot frozen ep0083 / 0.2618** at 02:59 (`ise-4090-10`). Path `/home/paretsky/scratch_audit/tree_v6_inband/runs/job21459737/snapshots/ep0083`. **FLAGS in SNAPSHOT_READY:** `SPECTRA_REWARD_SCALE=cbrt_cubes`, `SPECTRA_REWARD_MODE=structural`, profile `offline_train_v6_inband`, group_once=1. Probes **12/24/36/48/60/72 = 0.000**, then **ep84 = 0.2618** (r56-w6 0.166 / r20-w10 0.358). Train still R (ep89 at 05:15). **No afterok TEST child. Do not enqueue TEST until Ido says GO.** Freeze score ≈ V4/v3 0.262 probe ceiling that already TESTed as cloned-mild (§102 / §95) — the TEST is still the A/B (does linear in-band *walk* differently), not a claimed 2 pp win from the probe number. Do not overlay leap. Do not fill the 2-GPU hole with C-G DRL / Catalog L DRL / second linear. V4 **21394377 COMPLETED** 20 Sep 11:48 (ledger §110); freeze still ep0083 already TESTed — do not re-TEST.

**OPS DELTA 20 Sep 12:35 IDT — Ido sitting questions (accuracy-up / representation / 21459742 / Catalog L / Gilad matrix).** Confirm, tighten, or kill **§7** below in this sitting. Do not treat it as locked until you write. Do not overlay leap. Do not enqueue Catalog L DRL or C-G DRL. Linear `21459737` is still the only new-method GPU.

**OPS DELTA 20 Sep 09:40 IDT — overrides stale queue language below.** Do not treat §1.3–1.4 “PD / has not started / scancel tau6” as current.

- Tau6 **21394378 CANCELLED** 19 Sep 18:07 (Ido). Neonraw **21385161 COMPLETED** 19 Sep 23:28. Leap still `c08d513`. **Do not overlay leap.** Remaining R: v3-fpgm/svd/bnscale **21385158/59/60**, V4 **21394377**, ft40 **21443408**, in-band **21459737**. QOS **6/7**.
- **In-band-linear `21459737` is R** (~15h32, `ise-4090-10`). FLAGS confirmed `scale=cbrt_cubes`, structural, factored=0, 5-action fpgm, 24-net, 2-pass, group-cost, train FT 12/4, train_tau=10. **PPO 10:** `ev=0.627` `batch_score=0.270` `gap=+0.09`. Probes **ep12 / 24 / 36 all 0.000**. **No freeze. Do not TEST.** Telemetry GO (ev>0, gap>+0.05) is **not** a non-mild policy. Do not enqueue linear×C-G or linear×V4 until this job TESTs a freeze.
- Producers **21459742 COMPLETED** — ledger **§108**: r20 **−0.7 @ 0.988/0.980**, r56 **+0.0 @ 0.999/0.991**. Empty band, **≡ C-G §100**. Oral “maybe only producers matter” is not a recoverability win. **Do not start C-G DRL from this row.**
- C100 gate **21459732 COMPLETED** — ledger **§109**: **0 admits**. Emit **21459733 FAILED exit 2**. V6 C-G/C-G+ DRL **21459734/35 CANCELLED**. **P5-B3 is dead.** Do not mix unrecovered C100. P5-B2 fallback is a caption lock in this sitting, not a GPU today.
- V4 rewind **3/3** at ep240 (probe returned to 0.262, freeze still ep0083). fpgm and svd also used last rewind. ft40 probe24 still **0.241** = its only freeze.
- Fill Gilad note Q1–Q3 if still blank. Catalog L remains a **separate** paste: `docs/PROMPT_FABLE_CATALOG_L.md`.

**Status (ops): DO NOT OVERLAY leap `src/` while v3/V4 are R** (`21385158/59/60`, `21394377`; neonraw and tau6 are gone). Do not scancel those trains unless Ido says so in this sitting. Do not TEST v2c. No ImageNet DRL. Do not start a third ranking-menu train. Do not mix unrecovered C100 into a train set.

**This sitting is not V5-P8 again.** V5 delivered layer-replacement (default-off) and the no-agent recovery probe. Those TESTs are in. Ido’s GO (19 Sep 01:13) is **in-band-linear reward**, then a second look at the **CNN representation pipeline**. Layer-replacement DRL is **not** the headline cell.

**Separate sitting the same afternoon (do not mix into this GPU work):** Catalog L protocol lock — `docs/PROMPT_FABLE_CATALOG_L.md` and `docs/paper/CATALOG_L_TEST_PLAN.md`. Gilad 19 Sep: grocery list of nets is not an experimental setup. Ido may paste that prompt after or before this one in the same Fable chat. Do not start Catalog L DRL TESTs from the V6 sitting.

Paste into the existing thesis-mission overview chat (Fable 5.1 MAX), not a new tab, not Grok ops. Ops stays on Grok 4.6.

**Science-first.** Full-semester extension. Identifiable one-cell A/Bs.

---

## 0. What you delivered last sitting (18 Sep 01:50–04:00) — do not re-derive

Read, do not rewrite: `docs/PROMPT_FABLE_V5.md` §0e, `docs/V5_P8_RECOVERY_PROBE.md`, `docs/V5_SITTING_SUMMARY_18SEP.md`, `docs/paper/LOOP_ALGORITHMS.md` §9, ledger §98.

You implemented C-G / C-G+ default-off, P5-B3 + gate, P2-empty skip, P9 NEON-src corrections. Leap `src/` was never overlaid. Probes ran from `/home/paretsky/scratch_audit/tree`.

**Gilad note to fill (Ido will send him this file after you edit it):** `docs/paper/GILAD_LAYER_REPLACEMENT_19SEP.md`. Slots marked `_to fill before this note is sent to Gilad.` Fill **Q1 / Q2 / Q3 Fable (19 Sep)** in plain language (no P8/C-G jargon in that file). Do not invent Ido answers; his are already written.

---

## 1. Progress since your last effort (18 Sep 04:00 → 19 Sep 16:10)

Quote `[eval] TRAJ val_best` only. Yardstick = 2-pass group-once mild §93: r20 **−3.4 @ 0.536**, r56 **−6.6 @ 0.923**.

### 1.1 Frozen-actor TESTs (all cloned 2-pass mild keep except neonraw r56)

| Snap | Job | r20 | r56 | Read |
|---|---|---|---|---|
| v3-fpgm ep0011 | §95 `21428727` | −5.1 @ 0.536 | −6.8 @ 0.923 | cloned mild |
| v3-svd ep0011 | §97 `21433272` | −5.0 @ 0.536 | −6.7 @ 0.923 | ≡ fpgm |
| v3-neonraw ep0023 | §99 `21442936` | −4.1 @ 0.536 | −7.4 @ **0.757** | r56 did **not** clone keep |
| V4-factored ep0083 | §102 `21447387` | −3.7 @ 0.536 | −6.9 @ 0.923 | kindest *head*; still ≡ mild |
| v3-bnscale ep0011 | §107 `21447388` **today** | **−3.6 @ 0.536** | **−6.7 @ 0.923** | kindest *ranking* TEST; still ≡ mild |

None is a Gilad 2 pp family win. “First freeze ≡ mild” is **not** falsified. bnscale later froze **ep0107 / 0.262** (three snaps); that later snap is **untested** — do not auto-TEST it in this sitting.

### 1.5 Cluster now (21 Sep 11:31) — overrides §1.3–1.4

| Job | Arm | Now | Verdict |
|---|---|---|---|
| **21459737** | in-band-linear `cbrt_cubes` | R ep99, PPO-25; snaps **ep0083/0.2618** (TESTed §111) and **ep0095/0.2622** (same ceiling; do not auto-TEST); next probe 108 | keep running |
| **21512868** | in-band TRAJ ep0083 | **COMPLETED §111** 08:24 | r20 −3.5 @ 0.536; r56 **−7.1 @ 0.756** |
| **21385158** | v3-fpgm (control) | ep221, freeze **ep0011**, probe 216 = 0.251, rewind 3/3, PPO-55 `ev=−0.041` | keep as cubed control; spent |
| **21385159** | v3-svd | ep232, freeze ep0011, probe 228 = 0.208, rewind 3/3 | keep; weak |
| **21385160** | v3-bnscale | ep228, freeze **ep0107**, probe 228 = 0.241 (up from 0.210), rewind 3/3 | keep; later snap still untested |
| **21443408** | ft40 40/10 | ep61, **NEW freeze ep0059 / 0.2679** (beat 0.241 and 0.262 ceiling); next probe 72 | **keep.** TRAJ of ep0059 is a competing next-GPU — Fable locks. Do not auto-TEST |
| 21394377 | V4-factored | **COMPLETED** 20 Sep 11:48 §110 | do not re-TEST ep0083 |
| 21385161 / 21394378 | neonraw / tau6 | COMPLETED / CANCELLED | dead |

QOS **5/6** (`MaxTRESPU gres/gpu=6`). One hole. Leap `c08d513`. Producers **21459742 COMPLETED §108**. P5-B3 dead §109. Do **not** fill with C-G DRL.

### 1.2 No-agent layer-replacement probe (decision table is now a full row)

Same mild 2-pass walk; only recovery changes. Recipe A = keep leftover filters, full-net FT.

| Net | A (keep leftover) | C-G (throw-away, train group) | C-G+ (C-G + 0.1× polish) |
|---|---|---|---|
| skinny r20-w2 | **−3.4 @ 0.536** §93 | −0.9 @ 0.988 §100 | **−10.3 @ 0.884** §101 |
| skinny r56-w4 | **−6.6 @ 0.923** §93 | −0.1 @ 0.999 §100 | −0.2 @ 0.999 §101 |
| chenyaofo r56 94.37% | **−3.9 @ 0.661** §103 | −0.5 @ 0.999 §104 | **−1.0 @ 0.999** §106 |

**C-G+ ≪ A on all three nets.** Empty band wherever throw-away is the recipe, except skinny r20 C-G+ which cut a little and was 6.9 pp worse at a shallower size. 0 Tracebacks; 76/60 replacements actually ran. Group-budget median from thin C-G: **26.5** epochs.

Oral-reading ablation (`SPECTRA_FT_REINIT_SCOPE=producers`, thin C-G) **`21459742` COMPLETED §108** — r20 **−0.7 @ 0.988**, r56 **+0.0 @ 0.999**, ≡ C-G empty band. Consumers were **not** the failure. Do not start C-G DRL from this row.

### 1.3 Live trains (16:10 IDT, **stale — use §1.5**) — do not overlay leap

| Job | Arm | ep / probe / freeze | GPU verdict |
|---|---|---|---|
| **21394377** | V4-factored | ep 198, probe **192 = 0.262** (matched freeze after rewind 2/3 @ 188), freeze still ep0083 / 0.262, 4 snaps | **keep** — only live train whose probe returned to the freeze score |
| **21385160** | v3-bnscale | ep 150, probe 144 = 0.210, freeze ep0107 / 0.262, 3 snaps, critic `ev=−0.45` | **keep** — only v3 that re-froze later; TEST of first freeze is §107 |
| **21385158** | v3-fpgm | ep 143, probe 132 = 0.202, freeze stuck ep0011, rewind 2/3 | keep as the **control** for in-band-linear |
| **21385159** | v3-svd | ep 156, probe 156 = 0.210, freeze ep0011, rewind 2/3, `ev≈0` | weak; not a kill by yourself |
| **21385161** | v3-neonraw | ep 222, probe **identity**, freeze ep0023, rewind **3/3 used**, `ret_scale≈1e4` | **least worth the GPU** — cubes without cbrt, no new freeze |
| **21394378** | V4-tau6 | ep 160, probe **identity**, best **0.042**, no freeze, pmax 0.90 | **second-least worth the GPU** — τ=0.6 did not help |
| **21443408** | v5-ft40 | ep 11, FT **40/10** confirmed, recipe A, `cbrt` still on, 24-net | **keep** — isolated 12/4 vs 40/10 honesty cell; just started 06:25 |

### 1.4 Queue (QOS 7/7). First in line among PD

Slurm **priority** (not nice) decides the next GPU. Age beat nice this morning: ft40 (nice 60, queued 18 Sep) started at 06:25 when bnscale TRAJ ended, ahead of producers (nice 36) and in-band (nice 38).

**PD order now (higher Q first):**

1. **`21459742` producers thin C-G** — nice 36, Q 186 — next no-agent cell  
2. **`21459737` v6-inband-linear** — nice 38, Q 184 — **Ido’s GO train** (not started; no log yet)  
3. `21459732` C100 gate — nice 50  
4. `21459733` emit-admitted — afterok gate (exit 2 if <2 C100 → V6 DRL will not start)  
5. `21459734/35` V6 C-G / C-G+ DRL — afterok emit, C10-only until emit writes admitted.json  

**Ido: linear reward is highly prioritized.** Ops will not scancel. If you want in-band to start **today**, recommend Ido scancel **tau6 `21394378`** (and only if still hungry, neonraw `21385161`). That is Ido’s call in the sitting reply, not ops.

### 1.5 Infra that already exists for this sitting

- In-band-linear **code is in**: `SPECTRA_REWARD_SCALE=cbrt_cubes` in `src/utils.py` (`apply_reward_scale(..., cubed=)`). Live default `cbrt` unchanged. 22 unit tests green (`tests/test_reward_modes.py::test_cbrt_cubes_keeps_inband_linear`). ρ=20 → in-band **+20**, miss **−20**, gain **+20** (live cbrt: in-band **~+2.7**).
- Profile `offline_train_v6_inband` = v3-fpgm twin (24-net wide, 5-action fpgm, structural, 2-pass, group-cost, rewind, train FT 12/4) with **only** the scale changed.
- Job **`21459737`** submitted from **`/home/paretsky/scratch_audit/tree_v6_inband`** (copy of scratch + those files). Leap untouched.
- C100 candidates json is on scratch, leap `configs/`, and `/home/paretsky/spectra_v5_hold/`. Gate resubmit `21459732`.
- Gilad note: `docs/paper/GILAD_LAYER_REPLACEMENT_19SEP.md`.

---

## 2. Assigned this sitting (in order, one cell at a time)

### A. Fill the Gilad note (no GPU) — first

`docs/paper/GILAD_LAYER_REPLACEMENT_19SEP.md`: write **Fable (19 Sep, after TESTs)** under Q1–Q3. Keep that file’s outsider English. If your 19 Sep read differs from Grok’s, say so.

### B. Linear reward — **headline, already queued**

**Do not redesign the cell unless you find a bug in `cbrt_cubes`.** Confirm the math against §98 and your 18 Sep sitting Q1. Confirm `21459737` FLAGS when it starts (`SPECTRA_REWARD_SCALE=cbrt_cubes`, `SPECTRA_REWARD_MODE=structural`, `FT_REINIT_EDITED=0`). If the queued job is wrong, say so **before** it starts (it is still PD).

**Which infra to build it over (Grok’s recommendation, Ido asked you to confirm or override):**

The most promising *as-is* recipe for a **causal** linear-reward A/B is **live v3-fpgm `21385158`**: 24-net `database_offline_wide.json`, 5-action FPGM, `structural`+`cbrt`, 2-pass, group-cost, rewind, train FT 12/4, small Transformer, dropout 0. That is the control with a TESTed frozen snap (§95) and the longest-running ranking arm. **`offline_train_v6_inband` / `21459737` is already that twin.**

Do **not** put linear reward on:

- V4 factored head (two changes; V4 TEST still ≡ mild; second cell *after* `21459737` TESTs).  
- bnscale (keep-learning is real, but TEST of ep0011 is still mild; ranking is not the bottleneck).  
- C-G / C-G+ (see §3 — mix-up; decision table is No-P8-DRL from the no-agent walks).  
- P5-B3 catalog (gate has not admitted anyone).  
- tau6 / neon-raw.

If you disagree, write one paragraph and **one** alternative profile. Do not queue a second linear train in this sitting.

Optional (Ido GPU call, not a new method): scancel tau6 so `21459737` starts this afternoon.

### C. Representation pipeline — **second look, design this sitting, GPU later**

Encoder capacity already **failed to move r56-w4** under a near-uniform A2C policy (ledger **§16 LOCKED**: Transformer / set / wide / frozen BERT all ~−24 pp on r56-w4). **Do not reopen BERT as the default. Do not spend a GPU on encoder until linear reward has a TESTed non-mild policy.** Otherwise you will re-measure “encoder × mild clone.”

What Ido wants: is the representation **optimal**, can it be smarter / more sensitive, is there SOTA to draw from. Design **default-off** flags and a **one-cell** follow-up that runs **after** `21459737` has a snap worth TESTing. Unit tests on CPU. Do not overlay leap.

Grok’s read + suggested work (develop or kill with evidence; do not implement five things):

1. **Retest the encoder A/B only after a non-mild policy exists.** §16 is confounded by a uniform actor. The `set` encoder tying the Transformer on r56-w4 may mean “relational encoding is unused when the policy is mild,” not “Graphormer-style coupling is useless.”  
2. **Group-as-token, not layer-as-token.** SPECTRA prunes **channel groups**. Tokens are still **layers**. Coupling is a single learned scalar `block_affinity` on same-id pairs (`StateEncoder.py` ~130–134). Graphormer uses **distance buckets**; GPS / TokenGT add MPNN edges. A first-class **group token** (one token per prune unit) plus producer/consumer edge features (kernel, stride, residual-vs-concat) is the CNN-native next step, not a wider Transformer (`transformer_wide` already tied BERT on the hard net).  
3. **Shared actor/critic trunk.** `Actor` and `Critic` each subclass `Agent` and each own a `state_encoder` (`src/Model/Agent.py`). Value learning does not shape the policy encoder. PPO literature (shared torso, separate heads) is the cheap fix; ~halves agent compute. Default-off `SPECTRA_SHARED_ENCODER=1`.  
4. **Legal rates ignore the state** (audit 13 Sep): the mask is env width, not encoder output. Representation can change *which group to cut* and *how hard*, not which rates exist. Do not “fix” the encoder hoping to invent 0.7.  
5. **Stale moments.** Live refresh is the edited row’s span; downstream activation moments stay pre-edit unless `SPECTRA_REFRESH_ALL_FEATURES=1` (P8). A cheap ablation: full refresh under **recipe A** (no throw-away) — identifies “state lag” vs “throw-away.”  
6. **Probe-set bias.** Rewind/patience watch `resnet56-width6` + `resnet20-width10` only. A VGG/DenseNet probe pair is the representation-governor cell, not more Residual clones.  
7. **SOTA to steal inductive bias from, not to reimplement:** Graphormer spatial encoding; GPS (MPNN+Transformer); architecture-performance GNNs (NAP / GATES — *performance prediction*, not pruning; do **not** confuse with Michael Bohadana’s NAP2, still waiting on repo access); AMC’s per-layer embedding (weaker than relational); OFA/BigNAS supernet encodings (different problem). Frozen BERT remains the document’s proposal and §16’s negative.

Do **not** grow 24→48. Do **not** put per-filter tokens back (`BERT_INPUT_CRITIQUE.md` §4).

Canonical files: `src/BERTInputModeler.py`, `src/Model/StateEncoder.py`, `src/fortify.py` (`build_fortify_features`, `STATE_GROUPCOST_DIM`), `src/channel_groups.py`, `docs/BERT_INPUT_CRITIQUE.md`, ledger §16, audit 13 Sep map (legal mask, dual encoder).

### D. Layer-replacement DRL — **not this sitting’s train**

See §3. Keep `21459734/35` at the back. Do not start them by hand. Do not combine C-G+ with `cbrt_cubes` in one job.

### E. Git / overlay

Sitting files on a `v6` branch when Ido says. Overlay leap only when v3/V4 stop or Ido says. Reward train already has its own tree. Do not `git` the P8 scratch tree while `21443408` / gate / V6 PD/R.

---

## 3. C-G / C-G+: step in the right direction, or not, or too soon?

**No-agent answer (identified on the 90% walk):** throw-away recovery is **not** a step toward a better *heuristic* CNN walk. Keep-leftover (recipe A) recovers a real in-band cut; throw-away does not, including with polish, on three nets. That is the decision-table row “NEON-C looks dense-DNN-specific.”

**Learned-agent answer (not identified):** a DRL agent *could* refuse to cut groups that throw-away cannot recover, or cut less than 0.9. The no-agent walk **forces** 0.9 every legal group, so it cannot prove “an agent under C-G+ would clone never-prune.” It *does* prove that if the agent clones mild, C-G+ will look like §101/§106.

**Do not run C-G+ × linear-reward in one job.** That mixes the two remaining levers. Order: (1) linear reward on recipe A (`21459737`); (2) producers-scope no-agent (`21459742`); (3) only if (1) TESTs a **non-mild** policy **and** (2) shows producers ≈ A, consider C-G DRL as a later cell. Until then V6 C-G trains stay placeholders.

Grok: C-G+ is **not** the next DRL GPU. Ido queued the placeholders so we would not forget; that is not a GO to start them.

---

## 4. Fallacies / renovations Grok wants you to confirm or kill

1. **In-band `cbrt` is why every peaked policy clones mild** (§98 + every TESTed snap). Linear in-band is the test. If `21459737` still clones mild, the story is **not** the scale map (then: action menu, identity-skip of FT, or representation).  
2. **Encoder A/B under a uniform policy does not license “representation is solved.”** §16 LOCKED for BERT-as-default; open for “relational encoder × peaked policy.”  
3. **Dual actor/critic encoders** (above).  
4. **Train FT 12/4 vs TEST 40/10** is still untested; `21443408` is the cell — let it run. Do not caption 12/4 as equivalent.  
5. **Probe set = two thin ResNets** makes rewind a thin-keep governor.  
6. **Feature refresh span** vs full NEON `create_fe`.  
7. **Slurm age > nice** stole this morning’s hole for ft40. Not a science bug; do not fight it with extra submits.  
8. **Gain-arm ×2 (P2)** remains a no-op on 90–96% C10 nets. Do not implement.

---

## 5. Good news / bad news for the sitting (today)

**Good.** bnscale TEST is the kindest ranking snap (−3.6 / −6.7) even though it is still mild. V4’s probe **returned to 0.262** after rewind 2/3 — the governor is not dead. ft40 actually started (honesty cell). Catalog L recipe A recovered **−3.9 @ 0.661** on the 94.37% net. C-G+ executed cleanly (replacements, 0 TB) — the *code* works; the *recipe* does not recover CNNs on this walk. `cbrt_cubes` is implemented and queued. C100 json is no longer missing.

**Bad.** Every frozen actor still clones mild. C-G+ ≪ A on all three nets (skinny r20 polish was a disaster, not a near-miss). neonraw used rewind 3/3 and still probes identity. tau6 never froze. in-band-linear **has not started** (QOS + this morning’s age inversion). Gate has not run. Dual-encoder + group-as-layer tokens are untouched.

---

## 6. Do / don’t

**Do.** Fill Gilad note Q1–Q3 Fable slots. Confirm or override the v3-fpgm twin as the linear-reward base. Design (not GPU) the representation follow-up, default-off, after a non-mild snap. Recommend to Ido whether to scancel tau6 so `21459737` starts. Keep producers `21459742` in line.

**Don’t.** Overlay leap. Scancel v3/V4 yourself. Mix C-G+ with linear reward. Reopen BERT. TEST v2c. ImageNet DRL. Edit P8 scratch except emit writing `admitted.json` after the gate. Edit `SPECTRA_draft.md`. Quote wrap / `pass 1/1` / terminals over τ.

**Where to look**

| Need | Path |
|---|---|
| TESTs since V5 | ledger §§99–107; Fable V3 §8.4c–j |
| Reward math | `src/utils.py` `compute_reward` / `apply_reward_scale`; `tests/test_reward_modes.py`; ledger §98 |
| Queued linear job | `scripts/spectra.sbatch` `offline_train_v6_inband`; tree `/home/paretsky/scratch_audit/tree_v6_inband` |
| Layer-replacement TESTs | `docs/V5_P8_RECOVERY_PROBE.md` §A table; §§100–101, 103–104, 106 |
| Representation | `docs/BERT_INPUT_CRITIQUE.md`; `src/Model/StateEncoder.py`; `src/BERTInputModeler.py`; ledger §16; audit 13 Sep |
| Gilad note | `docs/paper/GILAD_LAYER_REPLACEMENT_19SEP.md` |
| Live trains | leap `runs/slurm_logs/spectra_21385158/9/60/61.out`, `21394377/78.out`; ft40 scratch `spectra_21443408.out` |
| Ido GO | this ops chat 19 Sep 01:13; `cbrt_cubes` already applied |

**Stamped:** 19 Sep 2026 16:15 IDT. Ops: Grok 4.6. Do not overlay leap.

---

## 7. Grok 20 Sep 12:35 — Ido’s questions for this sitting (confirm / kill)

Ido will paste this prompt next. Answer in the sitting reply **and** (if you agree) fold a short version into the Gilad note Q1–Q3 Fable slots. Do not mix this with a Catalog L GPU. Cluster at stamp: QOS **5/7**; V4 **21394377 COMPLETED** 11:48 (freeze already TESTed §102); in-band **21459737** mid-ep48, probes 12/24/36 = 0, **no freeze, do not TEST**.

### 7.1 Two different “empty” facts — do not fuse them

| Empty thing | What it is | Evidence | Not |
|---|---|---|---|
| **Gain arm** of the NEON trichotomy | Step reward branch `delta_acc > 0` after prune+FT vs origin | Ledger §98: **0 of ~17 700** non-identity *training* steps across seven trains. Gilad note §3: a `×2` bonus would not have fired. | C-G “empty band.” Representation. A silent crash. |
| **C-G empty band** | TRAJ `val_best` stays at identity (kept ≥ 0.98) because a forced 90% cut cannot be recovered | §§100–101, 104, 106, **108**. Same mild walk as recipe A, only recovery changes. | “Accuracy went up.” A DRL result. |

Recipe A **does** open an in-band *drop* (negative Δacc, inside τ): skinny r20 **−3.4 @ 0.536**, chenyaofo r56 **−3.9 @ 0.661**. The gain arm being empty means those recoveries never *overshoot the origin*. That is compatible with a useful pruner. NEON’s headline +0.5% is a different catalog and a different recovery.

### 7.2 Why the gain arm is empty — CNNs harder, or SPECTRA worse than NEON?

**Both, and they are already partly identified. Do not pick one slogan.**

1. **Catalog headroom (primary for the gain arm).** SPECTRA trains on already-strong CIFAR-10 CNNs (origin ~90–96%). Structured channel prune + **12-epoch Adam** is a regularizer only if the net is overfit. These nets are not. NEON’s +0.5% was on 28 datasets of *dense* nets with more slack. Acc-up on CNNs is not magically impossible: DepGraph Table 1 R56 is **93.53→93.64 (+0.11) at 2.57×** after **200-epoch SGD**. That is an existence proof under a different solver and a different FT. It is **not** a SPECTRA TEST, and it is **not** a reason to chase gain-arm ×2 (P2) on this catalog.

2. **Recovery translation (primary for the C-G empty band).** NEON source `create_new_model_with_new_weights` is native to a dense stack: new `Linear(in, k)`, new consumer `Linear(k, out)`, freeze the rest, train those modules. On a residual CNN the same idea redraws a group inside `F(x)+x` with frozen BN stats. The C-G *code ran* (76/60 replacements, **0 Tracebacks**). Failure is recipe × CNN, not a crash. SPECTRA’s live `--prune` (recipe A, keep leftover filters, full-net FT) **is** the method Gilad rejected for dense NEON — and it is the method that actually recovers CNN cuts. So: NEON was better *accustomed to DNNs*; SPECTRA is not uniformly “worse code.” The faithful port does not recover CNNs on this walk; the unfaithful keep-leftover port does.

3. **Train FT vs TEST FT.** Gain census is from **train** steps (12/4). TEST is 40/10. `21443408` (ft40) is the honesty cell — let it run. If ft40 also never records `delta_acc > 0`, the empty gain arm is not the 12/4 cost cut.

4. **C100 is not a free source of gain-arm slack.** P5-B3 gate **§109: 0 admits** under the *train* FT. Identity / densenet **−2.6 @ 0.989**. Weaker origin accuracy did **not** open an in-band band at 12/4. Do not mix unrecovered C100 to “create” acc-up.

**Representation does not empty the gain arm.** Δacc is measured after prune+FT in the env. The 17 700-step census includes heuristic walks (mild/L1) whose encoder is unused. If mild 90% + 40-ep FT on chenyaofo is **−3.9 not +0.1**, no encoder can invent a positive Δacc on that step. Representation can only *choose different groups/rates* — and only after the policy is no longer a mild clone.

### 7.3 Most flaky design area = the thesis novelty (representation), with a condition

Suspect novelty, in order, **after** the cells that already exist:

| Rank | Novelty | Status 21 Sep 05:17 | What would change the rank |
|---|---|---|---|
| **1 (flakiest, GPU later)** | **CNN state: layer tokens + skip PE + frozen-BERT-then-small-Transformer** (`BERTInputModeler` / `StateEncoder`). Dual actor/critic encoders. Legal rate mask ignores the state. Group is the prune unit; token is still the layer. Stale activation moments unless `REFRESH_ALL`. | Ledger **§16 LOCKED** under a near-uniform policy (Transformer / set / wide / BERT all ~−24 pp on r56-w4). **Not** re-identified under a peaked non-mild actor. | If in-band-linear **still clones mild** after the §7.7 freeze TEST, this becomes the next one-cell GPU (design now: group-as-token + shared trunk, default-off). If linear produces a non-mild policy, retest encoder on *that* actor before declaring representation solved or dead. |
| **2 (identified on the forced walk)** | **Throw-away recovery on CNN groups** (Gilad oral / NEON Algorithm 1 `compress_layer`). | C-G+ ≪ A on three nets; producers **§108 ≡ C-G**. | Only reopen if a *non-mild* actor exists **and** someone still wants C-G DRL. Do not mix with linear. Q3 (train-loss vs val plateau) still untested — that is a caption, not a GPU today. |
| **3 (froze; TEST pending Ido)** | In-band `cbrt` shrinking a legal cut to ρ^{1/3}. | **`21459737` froze ep0083 / 0.2618.** Probes 12–72 = 0; 84 = 0.2618. FLAGS `cbrt_cubes`. Same probe ceiling as V4/v3 that TESTed ≡ mild. | **§7.7 lock.** If TEST is non-mild, representation is no longer the first explanation of “always keep 90%.” If it still clones mild, kill the scale-map story and go to rank 1 / action menu / identity-skip of FT. |

**Do not reopen BERT as the default.** The document’s frozen `bert-base-uncased` is the *least* plausible CNN representation (English LM, frozen, 110M, dual forward). The live 3-layer Transformer is already the ablation winner vs BERT under a uniform policy. The remaining hypothesis is **relational / group-native tokens × a peaked policy**, not “put BERT back.”

**How to make acc-up *feasible* (science order, not a slogans list):**

1. Do **not** hunt acc-up as a GPU goal. In-band (legal drop, real cut) is the thesis. Gain is a bonus arm that is currently a no-op.
2. Finish **linear-reward** freeze TEST (`21459737`) — needed before any representation GPU, and before claiming the agent cannot beat mild.
3. Let **ft40** finish — needed before captioning 12/4 as the reason gains never happen.
4. Optional later, Catalog L protocol (separate paste): one **FLOP-matched** DepGraph-R56 row even if val leaves τ. That is the only cheap way to sit next to a published CNN *gain*. Do not enqueue it until (a) Fable locks the protocol and (b) a non-mild actor exists. Matched 200-ep SGD remains optional / later.
5. Skip P2 (`×2` on Δacc>0). Empty arm.
6. Skip C-G DRL. Skip mixing C-G+ × linear.

### 7.4 What `21459742` added (already COMPLETED — not a queue item)

No-agent mild 2-pass **C-G, producers-scope** (re-draw surviving **producer** filters + group norms; consumers sliced, not replaced; no polish). Gilad Q2 oral reading vs the NEON-source reading (also replace the consumer).

Ledger **§108:** r20 **−0.7 @ 0.988/0.980**, r56 **+0.0 @ 0.999/0.991**. Empty band, **≡ full-group C-G §100**.

**Benefit we already got:** Q2 is identifiable on the diagnostic pair. The empty band is **not** “we rebuilt too many consumers.” Throwing away the surviving producer filters is enough to fail recovery. Quote source-literal as the paper method; caption oral as an ablation that also failed. **Do not start C-G DRL from this row.** Heuristic only — no agent.

### 7.5 Catalog L — not `21459742`, not a GPU to enqueue soon

Two different objects:

| Object | Agent? | Status | What it buys |
|---|---|---|---|
| Catalog L **no-agent recovery** on chenyaofo r56 | No. Mild 90% walk. | **Already in:** A §103 **−3.9 @ 0.661**, C-G §104 empty, C-G+ §106 empty | Third net of the throw-away vs keep-leftover matrix. |
| Catalog L **protocol lock** (this afternoon’s other paste) | Neither heuristic nor DRL until you lock (a)(b)(c) | **Fable sitting, no GPU** — `docs/PROMPT_FABLE_CATALOG_L.md` | Thesis §4 setup: train vs test, metrics, budgets vs DepGraph’s CIFAR test set. Grocery list rejected 19 Sep. |
| Catalog L **DRL TEST** of a frozen actor on DepGraph L1–L3 | Yes, later | **Do not enqueue** | SOTA slide. Worthless until a **non-mild** actor exists (otherwise it is another mild-clone row on a home net). |

**Incentive to enqueue Catalog L compute soon: none.** The protocol sitting is ASAP (paper chapter). The GPU is after linear freeze TEST + locked protocol. Budget table (Gilad c: no per-target agent vs DepGraph search) can be *written* now without a job.

### 7.6 Gilad matrix — already exists; producers now included; claim strength

File: `docs/paper/GILAD_LAYER_REPLACEMENT_19SEP.md` §2 (English + Hebrew). Ops added the producers row 20 Sep. Decision table: `docs/V5_P8_RECOVERY_PROBE.md` §A. Ledger §§93, 100–101, 103–104, 106, 108.

**Allowed caption (PRELIM, no-agent, 90% walk):** NEON throw-away does **not** recover this CNN walk; SPECTRA keep-leftover **does**. C-G+ ≪ A on three nets. Producers-only ≪ A on the skinny pair. Show Gilad that table.

**Too soon / do not write as a validated thesis fact:** “the dense-DNN approach cannot transfer to CNNs.” Untested: DRL under C-G (walk *forces* 0.9, so an agent that refused to cut is not disproved); Q3 train-loss plateau; 200-ep SGD rematch; any catalog other than these C10 ResNets. Acc-up empty is a **separate** caption (catalog + short Adam), not a throw-away caption.

Fill Fable Q2 with this. Leave Q3 for Gilad. Do not edit `SPECTRA_draft.md`.

### 7.7 In-band ep0083 TRAJ — SUBMITTED 21 Sep 05:22 (no longer a lock)

Ido asked to fill empty slots. Ops treated that as GO. Job **`21512868` R** `traj-v6-inband-ep0083`.

**Grok 21 Sep 05:17 recommendation was GO.** The freeze is why we held the hole. Probe 0.2618 is not a win; the TRAJ is the A/B.

| | |
|---|---|
| Snap | `/home/paretsky/scratch_audit/tree_v6_inband/runs/job21459737/snapshots/ep0083` |
| Submit from | **scratch `tree_v6_inband`**, not leap (reward overlay lives there) |
| Profile | `eval_c10_thin_traj` — pin actor/critic/standardizer/policy_config from the snap; `SPECTRA_EVAL_DETERMINISTIC=1` `SPECTRA_EVAL_TRAJECTORY=1` `SPECTRA_SKIP_TRAIN=1` |
| FLAGS that must survive | `cbrt_cubes` / structural / group_once / 5-action fpgm menu / `STATE_ALIGN=next` (already in snap `policy_config.json` + `SNAPSHOT_READY.json`) |
| Quote | `[eval] TRAJ val_best` only. Skip r32. Never wrap / pass 1/1 |
| Compare | mild §93 (r20 **−3.4 @ 0.536**, r56 **−6.6 @ 0.923**); V4 same-ep §102 (r20 **−3.7 @ 0.536**, r56 **−6.9 @ 0.923**); v3-fpgm cubed control §95 |
| Do not | afterok on the live train; TEST from leap `src/`; TEST bnscale ep0107 or V4 again; Catalog L DRL; C-G DRL |

**Already submitted.** Train **21459737** stays R. Confirm FLAGS in the sitting reply; do not re-TEST V4 or bnscale ep0107.

### 7.8 Lock now — next GPU (QOS 5/6, one hole)

QOS cap is **6**. One GPU is idle **on purpose**. Do not invent a 7th job. `21512868` val_best **exists** (§111). §7.8 2a vs 2b is **mixed**: r20 cloned 0.536; r56 keep **0.756** (not 0.923). Not a 2 pp win. Catalog L DRL of this actor is **not** unlocked by a SOTA slide. Representation overlay is **not** forced by a full mild clone.

**Grok 11:31 rank (Fable may kill).** Pick **one**:

| Rank | Cell | Secure? | When |
|---|---|---|---|
| **1** | Skip-train TRAJ of **ft40 `21443408` snapshots/ep0059** (probe **0.2679**, beat 0.262 ceiling). From scratch `tree`, not leap. Same pin as §111 (`eval_c10_thin_traj`, det=1, recipe A). Compare §93 / §95 / §111 | Yes — isolated 12/4 vs 40/10 now has a freeze that *moved* | Cheapest identification. Ops will **not** submit until this sitting writes GO. |
| **2** | CNN representation one-cell (group-as-token + shared trunk, default-off). Not BERT. | Design now; GPU after overlay is in git | Rank-1 novelty. Do not start from ops. |
| **3** | Catalog L **DRL** of in-band ep0083 on DepGraph L1 — only after Catalog L §5 is filled **and** Fable argues the mixed keep is enough | Protocol sitting still blank | Not Grok’s fill of this hole. |
| **no** | C-G / C-G+ DRL; P5-B3; ImageNet DRL; second linear × C-G or V4; bnscale ep0107; in-band ep0095 auto-TEST; release JobHeldUser heuristics; overlay leap | Unsafe or spent | |

**Write 1 or 2 or WAIT (or a §7.9.2 replacement) in this sitting.** Do not mix. Experiment-plan canvas is dropped.

### 7.9 Inventing a non-mild actor — wait vs sitting now; C-G-for-DRL; next directions (Fable must ponder, may kill)

Ido (21 Sep 05:28): (1) do we have enough to feed Fable *now* to invent the first actor that does not clone mild, or wait ~7 h for live jobs? (2) C-G / C-G+ are crossed for the *heuristic* walk — can we be sure they are a bad approach for **DRL training**? (3) what next to explore? Leave room for Fable beyond Grok’s pre-existing list.

**Grok’s wait/go (confirm or kill):** **Wait.** Do **not** overlay leap. Do **not** start a representation / action-menu / C-G DRL train from this sitting if `21512868` has no `[eval] TRAJ val_best` yet. Typical thin TRAJ is ~5 h; Ido is back in ~7 h; ops will ledger the TEST. One sitting *with* that row is worth more than a sitting that invents against a cell still in flight.

What we **do** have enough for *without* that row (paper + design, no GPU):

- Catalog L protocol lock (`docs/PROMPT_FABLE_CATALOG_L.md` §5) — Gilad 19 Sep, independent of the actor.
- Gilad note Q1–Q3 Fable slots (`docs/paper/GILAD_LAYER_REPLACEMENT_19SEP.md`).
- Default-off overlay **design** for the next one-cell train (write flags + tests; do not submit).
- The C-G-for-DRL argument below.

What we **do not** have: whether linear-in-band already walks off 90%. Inventing representation now, if `21512868` is non-mild, wastes the sitting on the wrong lever. Inventing it now, if the clone is actually “identity is free + 0.9 is the only legal cut,” wastes it on the encoder.

#### 7.9.1 Is C-G / C-G+ a bad *DRL training* recipe? (not sure — argue)

**Identified (no-agent, forced 90% walk, recipe is the only change):** throw-away does not recover that walk. Keep-leftover does. Producers-only ≡ full-group C-G. Polish does not save it. Code ran (0 TB). This is **recovery × CNN**, not a crash.

**Not identified:** a learned agent under C-G as the env’s `--prune` recovery. The no-agent walk *forces* 0.9 on every legal group, so it cannot prove “an agent would also refuse to cut.” An agent *could* pick identity, or 0.8, or different groups.

**Grok prediction if you train DRL with C-G anyway:** every 0.9 cut that fails recovery dumps a large negative reward → the rational policy is **identity** (rate 1.0). That would make cloning *worse*, not produce the first non-mild actor. Recipe A is the only recovery that currently *pays* for a real cut.

**Remaining sliver (Fable should keep or kill):** the agent might choose *easier groups / gentler rates* that throw-away can recover, even though forced 90% on these ResNets cannot. Testing that still mixes two novelties (new recovery × new policy). Order if you insist: a **non-mild recipe-A actor first**, then a no-agent C-G walk *of that actor’s groups*, not a C-G DRL train from a mild clone.

**Q3 still open:** train-loss plateau vs val plateau (Gilad oral). Caption, not a GPU, until someone wants it.

**Fable: write whether you agree that C-G DRL is not the next train**, or name the one identifiable cell that would change your mind. Do not rubber-stamp “skip C-G DRL” in one clause without the sliver.

#### 7.9.2 Next directions (Grok rank — Fable may reorder, add, or kill)

Pre-existing (already in §2C / §4 / §7.3): group-as-token; shared actor/critic trunk; legal mask vs state; stale moments / `REFRESH_ALL` under recipe A; probe-set bias (VGG/DenseNet governor); Graphormer/GPS inductive bias (not BERT); action menu; identity-skip of FT; linear-in-band (in flight).

**After `21512868` val_best, Grok would explore in this order. Fable: keep, swap, or replace. Pick **one** GPU cell, not a menu.**

| # | Direction | Why it could produce a non-mild actor | Why it might not | GPU now? |
|---|---|---|---|---|
| 0 | **Read `21512868`** | Linear-in-band was the last *reward-side* hypothesis for “peaked ⇒ 90%.” | **DONE §111.** r20 cloned keep; r56 left 90% (≡ neonraw). Scale-map dead on r20, mixed on r56. | COMPLETED. |
| 1 | **Group-as-token + shared trunk** (default-off) | Prune unit ≠ token; dual encoders; legal mask ignores state. Thesis novelty. §16 confounded by uniform policy. | Encoder A/B already tied under uniform. Relational encoding unused if the head is a 0.9 bias. | Design now; submit only if #0 still clones mild. |
| 2 | **Identity-skip of FT / don’t reward identity as a step** | A peaked policy can look decisive while still being “0.9 on every legal group, 1.0 elsewhere.” If identity is free, PPO has no reason to leave mild. | May already be mostly true (identity steps are ~1 s). Changing the MDP without a new representation can still clone 0.9. | One-cell, default-off, **xor** with #1. |
| 3 | **Action menu: add 0.7 (or drop 0.9 as the “easy” cut)** | Legal mask cannot invent a rate that is not in the menu. 1.0/0.9/0.8 makes 0.9 the interior mild. | Another ranking-menu train. Cadence: do not start another ranking-menu train. Only reopen if Fable argues 0.9-*availability* is the clone. | Only if Fable writes why this is not ranking-menu redux. |
| 4 | **Catalog L DRL of a *non-mild* actor on DepGraph L1** | Gilad: sit next to SOTA on their net; FLOP-matched even if val leaves τ. | Worthless if the actor is still mild (home-net mild-clone). Needs §5 protocol. | After #0 is non-mild **and** Catalog L §5 filled. |
| 5 | **ft40 honesty** | Gain-arm empty might be 12/4. TEST FT is 40/10. **NEW freeze ep0059 / 0.2679 beat the 0.262 ceiling.** | Probe 48 was 0.180; the later freeze is the cell. TRAJ not run. | Already R. Next GPU candidate = TEST of ep0059, not a twin. |
| 6 | **Full feature refresh under recipe A** | State lag: downstream activation moments stay pre-edit. | Cheap; might be a caption not a train. Can ride along as a flag on #1. | Flag on #1, not a separate train. |
| **no** | C-G / C-G+ DRL; P5-B3; ImageNet DRL; BERT-as-default; second linear×C-G; P2 (`×2` on Δacc>0); grow 24→48; per-filter tokens | See §7.9.1 / Gilad / §16 | | |

**Fable deliverable if pasted before TRAJ:** (i) C-G-for-DRL write-up for Gilad Q-slots; (ii) one default-off overlay spec (#1 or #2 or a *new* idea you invent — argue it); (iii) explicit **do not submit** until ops pastes `21512868` val_best into an OPS DELTA. If you invent a direction not in the table, name the one-cell A/B and the kill criterion.

**Fable deliverable if pasted after TRAJ:** lock §7.8 2a vs 2b vs a §7.9.2 replacement; then Ido/ops submits.

Do not edit `SPECTRA_draft.md`. Do not overlay leap. Do not start Fable’s overlay from ops.

**3-hour ops briefing cadence (21 Sep):** every ~3 h Grok writes good/bad/insight in the ops chat **and** appends an OPS DELTA here. Current-affairs still stamps 09:30 / 16:00 / 23:00 files-only.

---

## 8. Fable 21 Sep 17:20–18:40 — sitting answers and locks

Ido's paste was truncated at §2 ("P5-B2 (keep one SVHN; hold FM…"); everything after that was reconstructed from this file, the Catalog L paste and the ledger. If the tail asked for anything not covered below, paste it again.

### 8.1 §7.8 LOCK — **GO rank 1, already submitted**

**`21535193` `traj-v5-ft40-ep0059`** — R since 17:36 on `cs-pheno-05`, from scratch `tree` (where ft40 runs), `eval_c10_thin_traj`, det=1, TRAJ, 2-pass, snapshot pins verified (`policy_config`: passes 2, 5-action fpgm, `ft_recipe=A`, train FT **40/10**, `cbrt`, group-once, align next, group-cost). Ops: **do not resubmit.** Log `/home/paretsky/scratch_audit/tree/runs/slurm_logs/spectra_21535193.out`; ledger it as **§112** vs §93 / §95 / §111 / §99. Nothing cancelled; QOS 6/6.

Why rank 1 and not 2: the ft40 freeze is the only selection score above the 0.262 ceiling every 12/4 arm shares, and the isolated FT A/B is the honesty cell every v2–V6 caption depends on. Representation (rank 2) has no identification yet (§8.4) and needs a 7-day slot, not a 4-hour hole. Catalog L DRL (rank 3): protocol is now locked (§8.5) but no actor is clean on L1/L2 and none is a win.

**Fill order for the next holes (ops submits when a GPU frees; not a 7th job now):**

1. **Matched-keep heuristic controls on thin, 3 passes** — `SPECTRA_EVAL_PASSES=3` `baseline_c10_mild_traj_gonce` and `baseline_c10_l1_traj_gonce` (recipe A, 40/10, det TRAJ). Needed to caption the two 0.756-keep r56-w4 rows (§99 neonraw, §111 in-band) and whatever `21535193` returns: today **no heuristic reaches ~0.75 kept in band** on r56-w4 (mild stops 0.923, L1 leaves the band past 0.898). If 3-pass heuristics cannot either, that gap *is* the learned-schedule sentence; if they can at ≤ 0.5 pp worse, it is not.
2. **Catalog L twins, same-loop controls** — 2-pass mild and L1 on `configs/input_catalog_l_twins.json` (chenyaofo r56 C10 mild A exists §103; L1 missing; VGG-16 C10 and VGG-19 C100 both missing). Bar 2 of the lock needs them; no agent, any hole.
3. **Next DRL train** when a v3 arm ends: `offline_train_v6_inband_p5b2` (§8.6). Not before (1) has a job id — the heuristics are cheaper and gate the caption.

### 8.2 §7.1–7.2 — the empties (confirm, plus one)

Confirm both empties and keep them apart. Add a **third**: skinny **r20-w2 is discrimination-empty**. All nine actor/heuristic TESTs since §93 land on the identical `0.536/0.655` walk because its streams are 2/4/8 channels wide and most groups have one realisable cut. Keep it as the "does it cut" sanity row; stop reading it as a policy comparison (written into Catalog L §5 and the Gilad note Q1).

Why the gain arm is empty — both causes, and now the third is closed: **ft40 (40/10) also has 0 gain steps** (2649 non-identity, max Δacc −0.80 pp); in-band-linear 0 of 3477 (ledger §98 addendum). Not the cost cut, not the cube-root; catalog headroom + short Adam. P2 stays dead.

### 8.3 §2B — linear reward: no bug; the cell did what the arithmetic said, on the net that can move

`apply_reward_scale(..., cubed=)` verified: `cbrt_cubes` → in-band **+ρ**, miss **−ρ**, gain **+ρ**; live `cbrt` untouched. §111 is the mechanism story confirmed on r56-w4: the two actors whose in-band arm is linear (raw cubes §99, cbrt_cubes §111) are exactly the two that left the 0.923 keep (0.757 / 0.756, in band). Every cbrt actor cloned mild there. Base recipe (v3-fpgm twin) was right; do not put linear on V4 / bnscale / C-G / C100. **Linear in-band becomes the default reward for new trains** (§8.6); the running v3 arms stay as cubed controls.

Telemetry caveat: `gap_to_uniform` +0.016–0.018 says the in-band policy is close to uniform over legal rates while its argmax walk is different — the argmax, not the peakedness, carries the schedule. This is why §8.4 starts with "is the state read at all", not with a bigger encoder.

### 8.4 §2C / §7.3 / §7.9.2 — representation: identify first (tool built), one cell specified, no GPU locked

`docs/V6_REPRESENTATION_DESIGN.md`. Built and unit-tested (default off, CPU 269/269 green in `tree_v6_dev`): **`SPECTRA_EVAL_COUNTERFACTUAL=1`** — at each actor step the runner logs whether the argmax changes when layer features are zeroed / shuffled across positions / everything but position-type-marker is zeroed. Rides on any actor TRAJ for free; answers "does the frozen policy read the state" — the number §16 never had. Not on any scratch tree yet; `21535193` does not carry it. Overlay `src/fortify.py` + `a2c_agent_reinforce_runner.py` onto `tree_v6_inband` / `tree` when no job is *starting* from them; use it on the next actor TRAJ.

The one cell, if the probe says per-layer content is read and the next actor still does not beat the heuristics at matched size: **group-as-token + relational attention bias** (`SPECTRA_STATE_TOKENS=groups`), shared actor/critic trunk as a *separate* second cell. Killed: BERT default, wider Transformer, per-filter tokens, 0.7 in the menu (the in-band actor reached 0.756 with {1.0, 0.9, 0.8} — the clone was the reward, not the ladder), "identity-skip of FT" (subsumed by linear in-band: identity now has an opportunity cost). Probe set for new profiles is VGG-13 + r56-w6.

### 8.5 Catalog L — §5 LOCKED (separate paste, done in this sitting)

`docs/paper/CATALOG_L_TEST_PLAN.md` §5 + §6 (thesis §4.1 draft). Anchor = DepGraph CIFAR test set on **their** checkpoints (L1 R56 93.53, L3 VGG-19 C100 73.5) + OCS L2 VGG-16 C10; two operating points (τ-matched, size-matched); measured budgets; three ordered "better" bars; matched 200-ep FT later/optional. **VGG-16 C10 leaves the next train** (probe → VGG-13; configs + tests updated). 24-net actors are in-catalog on L1/L2 (weights unseen), clean only on L3. DepGraph ckpts need a **CPU loadability check** before any TEST (ops: `init_catalog_l.py` / compat check; they are `reproduce` objects). LOOP §7 rewritten; `catalog_l_map.json` flags set.

### 8.6 Next DRL train — profile ready, not submitted

`offline_train_v6_inband_p5b2` = in-band linear reward (§8.3) × Catalog-L-clean catalog `configs/database_offline_v6_p5b2.json` (9-net C10 core with VGG-13 + **one SVHN net**, VGG-11 SVHN; P5-B2 because C100 admitted nothing §109) × probes VGG-13 + r56-w6 × train FT 12/4. Disjointness unit-tested (train ∩ {L1–L3, thin, similar, unlike, FMNIST, ImageNet, remaining SVHN} = ∅). **Submit with `SPECTRA_TRAIN_FT_EPOCHS=40 SPECTRA_TRAIN_FT_PATIENCE=10` only if `21535193` shows the 40/10 freeze walks differently from the 12/4 arms** (deeper keep in band, or ≥ 1 pp kinder at equal keep on r56-w4); otherwise 12/4. That is the thesis actor for Catalog L bars 1–2. From `tree_v6_dev` (has cbrt_cubes + this sitting's files) — ops copies it to a clean `tree_v6_train` before submitting so the dev tree can keep moving.

### 8.7 §7.9.1 — C-G / C-G+ for DRL: not the next train; the one cell that would reopen it

Agree it is not next. The sliver is real but small: an agent could restrict itself to cuts throw-away can recover. The **identifiable cell** that would change my mind costs one heuristic job, no training: replay the in-band actor's r56-w4 action sequence (its 0.756 walk, step records of `21512868`) under recipe C-G+. If C-G+ recovers *that* walk within ~1 pp of A, a C-G DRL train is plausible; if it goes empty-band like §101/§106, NEON-C is closed for CNNs. Needs a ~40-line `SPECTRA_EVAL_POLICY=replay` mode (not written this sitting). Until someone wants it: no C-G GPU, and never mixed with the linear reward.

### 8.8 Gilad note — Q1–Q3 Fable slots filled (`GILAD_LAYER_REPLACEMENT_19SEP.md`)

Outsider English, no P8 jargon: Q1 = arithmetic confirmed on the net that can move (§111), r20 retired as a comparison, recommend linear in-band as default for new trains; Q2 = producers-only ≡ group, throw-away of surviving filters is itself the failure, budget was not the problem (median plateau 26 of 60 epochs), no C-G training GPU; Q3 = quote both rules in one sentence, validation as SPECTRA's, train-loss rule cannot rescue §2.

### 8.9 Not done this sitting / needs Ido

- Anything in the truncated tail of the paste after §2.
- Ido signs Catalog L §5 (and the VGG-16 hold-out); ops pastes `docs/PROMPT_OPS_V6_LOCK.md`.
- Overlay of the counterfactual probe onto the scratch trees (ops, when safe); git `v6` branch when Ido says.
