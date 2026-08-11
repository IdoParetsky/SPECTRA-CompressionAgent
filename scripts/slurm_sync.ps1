#!/usr/bin/env pwsh
# Sync committed SPECTRA code from this PC to the BGU cluster home clone.
# Usage:
#   .\scripts\slurm_sync.ps1
#   .\scripts\slurm_sync.ps1 -HostAlias bgu-slurm-gpu
param(
    [string]$HostAlias = "bgu-slurm",
    [string]$RemoteRepo = "/home/paretsky/SPECTRA-CompressionAgent",
    [switch]$SkipPush
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

$status = git status --porcelain
if ($status) {
    Write-Host "WARNING: local working tree has uncommitted changes. Only committed commits will sync via git." -ForegroundColor Yellow
    Write-Host "Commit (or stash) locally if you need those edits on the cluster." -ForegroundColor Yellow
}

if (-not $SkipPush) {
    $ahead = git rev-list --count "origin/master..HEAD" 2>$null
    if ($ahead -and [int]$ahead -gt 0) {
        Write-Host "Pushing $ahead local commit(s) to origin/master..."
        git push origin HEAD:master
    } else {
        Write-Host "Local master not ahead of origin (or already pushed)."
    }
}

$remoteScript = @"
#!/usr/bin/env bash
set -euo pipefail
cd '$RemoteRepo'
echo "Remote before: `$(git rev-parse --short HEAD) (`$(hostname)`)"
if [[ -n "`$(git status --porcelain)" ]]; then
  git stash push -u -m "cursor-slurm-sync `$(date -Iseconds)" || true
  echo "Stashed remote dirty state (if any)."
fi
git fetch origin
git checkout master
git pull --ff-only origin master
echo "Remote after:  `$(git rev-parse --short HEAD)"
test -f scripts/check_gpu.sh && test -f scripts/run_agent.sh && echo "Helper scripts present."
"@
$remoteScript = $remoteScript -replace "`r`n", "`n" -replace "`r", "`n"

$tmpLocal = Join-Path $env:TEMP "spectra_slurm_sync.sh"
$tmpRemote = "/tmp/spectra_slurm_sync_$env:USERNAME.sh"
[IO.File]::WriteAllText($tmpLocal, $remoteScript, (New-Object System.Text.UTF8Encoding $false))

Write-Host "Pulling on ${HostAlias}:$RemoteRepo ..."
scp -o BatchMode=yes $tmpLocal "${HostAlias}:$tmpRemote"
ssh -o BatchMode=yes $HostAlias "bash $tmpRemote; rm -f $tmpRemote"
Remove-Item -Force $tmpLocal -ErrorAction SilentlyContinue

# Keep cluster helpers present even before they are committed / after remote stash -u
$helpers = @(
    (Join-Path $repo "scripts\check_gpu.sh"),
    (Join-Path $repo "scripts\run_agent.sh"),
    (Join-Path $repo "scripts\run_spectra_sinteractive.sh")
) | Where-Object { Test-Path $_ }
if ($helpers.Count -gt 0) {
    foreach ($f in $helpers) {
        $c = [IO.File]::ReadAllText($f) -replace "`r`n", "`n" -replace "`r", "`n"
        [IO.File]::WriteAllText($f, $c, (New-Object System.Text.UTF8Encoding $false))
    }
    ssh -o BatchMode=yes $HostAlias "mkdir -p '$RemoteRepo/scripts'"
    scp -o BatchMode=yes $helpers "${HostAlias}:$RemoteRepo/scripts/"
    ssh -o BatchMode=yes $HostAlias "chmod +x '$RemoteRepo'/scripts/*.sh"
    Write-Host "Redeployed helper scripts to remote scripts/."
}

Write-Host "Sync done."
