# Overnight: wait for recover_king, summarize matrix, submit recover_careful, write morning brief.
$ErrorActionPreference = "Continue"
$Repo = "C:\SPECTRA-CompressionAgent"
$King = "20061144"
$Wander = @("20053619", "20053621", "20053625")
$BriefLocal = Join-Path $Repo "runs\morning_brief.txt"
$Sentinel = "AGENT_LOOP_WAKE_overnight_king"

Set-Location $Repo
function Rexec([string]$Cmd) {
  & .\scripts\rexec.ps1 -Command $Cmd 2>&1 | Out-String
}

Write-Output "$Sentinel started waiting for king=$King"

while ($true) {
  $st = (Rexec "sacct -j $King -n -X -o State --parsable2 | head -1").Trim()
  if ($st -match "COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPTED|OUT_OF_MEMORY|NODE_FAIL") {
    Write-Output "$Sentinel king_terminal state=$st"
    break
  }
  $tail = Rexec "grep 'DONE Episode' /home/paretsky/SPECTRA-CompressionAgent/runs/slurm_logs/spectra_$King.out 2>/dev/null | tail -2"
  Write-Output "$Sentinel waiting king=$King state=$st || $tail"
  Start-Sleep -Seconds 900
}

# Sync code that may have been committed before sleep, then submit careful.
Rexec "cd /home/paretsky/SPECTRA-CompressionAgent && git fetch origin master && git reset --hard origin/master" | Out-Null

$summary = Rexec @"
cd /home/paretsky/SPECTRA-CompressionAgent
echo '==== KING summarize ===='
python3 scripts/summarize_run.py runs/job$King 2>&1 | tail -n 80
echo '==== non-id within -10pp (king) ===='
python3 - <<'PY'
import json,glob,statistics
from pathlib import Path
ev=Path('runs/job$King/events')
rows=[]
for f in ev.glob('*.jsonl'):
  for line in f.open():
    o=json.loads(line)
    if o.get('event')!='step': continue
    rate=o.get('compression_rate') or o.get('rate')
    if rate is None: continue
    try: rate=float(rate)
    except: continue
    if abs(rate-1.0)<1e-9: continue
    dacc=o.get('delta_acc', o.get('acc_delta', o.get('accuracy_delta')))
    rew=o.get('reward')
    rows.append((rate,dacc,rew,o.get('episode')))
print('non_id_steps', len(rows))
if rows:
  within=[]
  for r,d,rw,ep in rows:
    if d is None: continue
    within.append(abs(float(d))<=0.10 or float(d)>=-0.10)
  # delta usually negative for drop
  ok=[(d is not None and float(d)>=-0.10) for _,d,_,_ in rows]
  print('non_id_within_m10pp', sum(1 for x in ok if x), '/', len(ok))
  ds=[float(d) for _,d,_,_ in rows if d is not None]
  if ds: print('median_delta_acc', statistics.median(ds))
PY
for j in $King $($Wander -join ' '); do
  echo ==== job\$j ====
  sacct -j \$j -n -X -o JobID,State,Elapsed,ExitCode --parsable2 | head -1
  grep -E 'DONE Episode|Stopping training|finished with status|FATAL' runs/slurm_logs/spectra_\$j.out 2>/dev/null | tail -5
done
echo '==== submit recover_careful ===='
bash scripts/submit.sh recover_careful
squeue -u paretsky
"@

$carefulId = ($summary | Select-String -Pattern 'submitted job (\d+)' | Select-Object -Last 1)
if ($carefulId) { $cid = $carefulId.Matches[0].Groups[1].Value } else { $cid = "UNKNOWN" }

$brief = @"
SPECTRA morning brief ($(Get-Date -Format o))
========================================
King job $King ended: $st
Follow-up recover_careful: $cid (12h wall, 6-net DB, warmup floor 100, standardizer ON, -10pp, rates 1.0/0.9/0.8)

Remote summary excerpt:
$summary
"@
Set-Content -Path $BriefLocal -Value $brief -Encoding UTF8
Rexec "cat > /home/paretsky/SPECTRA-CompressionAgent/runs/morning_brief.txt << 'EOF'
$brief
EOF
"

Write-Output "$Sentinel done careful=$cid brief=$BriefLocal"
Write-Output $summary
