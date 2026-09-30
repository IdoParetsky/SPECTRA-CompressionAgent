#!/usr/bin/env bash
# Read-only pre-flight for the P area train on tree_v9c (one change vs area train 21536396 on tree_v7).
V7=/home/paretsky/scratch_audit/tree_v7
C=/home/paretsky/scratch_audit/tree_v9c
echo "=== sbatch diff tree_v7 -> tree_v9c (profile block lines only)"
diff <(sed -n '/offline_train_v6_inband_p5b2/,+3p' $V7/scripts/spectra.sbatch) <(sed -n '/offline_train_v6_inband_p5b2/,+3p' $C/scripts/spectra.sbatch) && echo "profile block lines identical"
diff $V7/scripts/spectra.sbatch $C/scripts/spectra.sbatch | grep -E '^[<>]' | grep -vE '^\s*[<>]\s*#' | wc -l
diff $V7/scripts/spectra.sbatch $C/scripts/spectra.sbatch | grep -E '^[<>]' | grep -vE '^[<>]\s*#' | cut -c1-160 | head -40
echo "=== submit.sh diff (non-comment lines)"
diff $V7/scripts/submit.sh $C/scripts/submit.sh | grep -E '^[<>]' | grep -vE '^[<>]\s*#' | cut -c1-160 | head -20
echo "=== catalog + syntax"
ls -la $C/configs/database_offline_v6_p5b2.json && cmp $V7/configs/database_offline_v6_p5b2.json $C/configs/database_offline_v6_p5b2.json && echo "catalog identical to tree_v7"
bash -n $C/scripts/spectra.sbatch && bash -n $C/scripts/submit.sh && echo "syntax ok"
echo "=== area train 21536396 policy_config (control)"
P=$(ls $V7/runs/job21536396/agent_checkpoints/policy_config.json 2>/dev/null); echo "$P"
[[ -n "$P" ]] && python3 -c "import json,sys; d=json.load(open('$P')); print({k: d[k] for k in ('compression_rates','action_rankings','factored_head','token_feature_dim','ft_recipe','passes')}); print('env', d['env']); print('info', d['info'])"
grep -m3 -E "FLAGS|GPU|gres" $V7/runs/slurm_logs/spectra_21536396.out 2>/dev/null | cut -c1-300
sacct -j 21536396 -X -n -o State%12,Elapsed,NodeList%16,AllocTRES%60 2>/dev/null | head -1
echo "=== group-token 21716380 files"
ls -la /home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/ 2>&1 | head -12
ls -d /home/paretsky/scratch_audit/tree_v8b/runs/job21716380/snapshots/* 2>&1 | head -5
ls -la /home/paretsky/scratch_audit/tree_v8b/runs/job21716380/agent_checkpoints/ 2>&1 | head -8
