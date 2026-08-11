# Overnight: wait for recover_king, summarize, submit careful A/B (+ king_fortify), write morning brief.
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

Rexec "cd /home/paretsky/SPECTRA-CompressionAgent && git fetch origin master && git reset --hard origin/master" | Out-Null

$summary = Rexec @"
cd /home/paretsky/SPECTRA-CompressionAgent
echo '==== KING summarize ===='
python3 scripts/summarize_run.py runs/job$King 2>&1 | tail -n 80
echo '==== submit A/B careful + careful_fortify (king_fortify may already be running) ===='
bash scripts/submit.sh recover_careful
bash scripts/submit.sh recover_careful_fortify
# Only submit king_fortify if none is already queued/running
if ! squeue -u paretsky -n spectra-recover_king_fortify -h | grep -q .; then
  bash scripts/submit.sh recover_king_fortify || true
fi
squeue -u paretsky
for j in $King $($Wander -join ' '); do
  echo ==== job\$j ====
  sacct -j \$j -n -X -o JobID,State,Elapsed,ExitCode --parsable2 | head -1
  grep -E 'DONE Episode|Stopping training|finished with status|FATAL' runs/slurm_logs/spectra_\$j.out 2>/dev/null | tail -3
done
"@

$ids = [regex]::Matches($summary, 'submitted job (\d+)') | ForEach-Object { $_.Groups[1].Value }
$brief = @"
SPECTRA morning brief ($(Get-Date -Format o))
========================================
King job $King ended: $st
Submitted follow-ups (careful A/B +/- fortify, king_fortify): $($ids -join ', ')

Fortify package (SPECTRA_FORTIFY=1): stem/narrow action mask, entropy anneal, depth/stem/coupling/width tokens, train_resume.pt mid-run bundles.

Remote excerpt:
$summary
"@
Set-Content -Path $BriefLocal -Value $brief -Encoding UTF8
Rexec "cat > /home/paretsky/SPECTRA-CompressionAgent/runs/morning_brief.txt << 'EOF'
$brief
EOF
"

Write-Output "$Sentinel done submitted=$($ids -join ',') brief=$BriefLocal"
Write-Output $summary
