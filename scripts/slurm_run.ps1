#!/usr/bin/env pwsh
# Sync (optional) then run SPECTRA on the allocated GPU node via SSH.
# Resolves the current sinteractive node automatically (popup if needed).
# Tees the full run to runs/slurm_logs/ for Cursor debugging.
#
# Usage:
#   .\scripts\slurm_run.ps1
#   .\scripts\slurm_run.ps1 -SkipSync
#   .\scripts\slurm_run.ps1 -PromptNode
#   .\scripts\slurm_run.ps1 -Node cs-1080-05
#   .\scripts\slurm_run.ps1 -ExtraArgs "--passes","1"
param(
    [string]$GpuHostAlias = "bgu-slurm-gpu",
    [string]$RemoteRepo = "/home/paretsky/SPECTRA-CompressionAgent",
    [string]$Node = "",
    [string]$LogDir = "",
    [switch]$PromptNode,
    [switch]$SkipSync,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $scriptDir

$resolveArgs = @{
    GpuHostAlias = $GpuHostAlias
    Quiet         = $true
}
if ($Node) { $resolveArgs.Node = $Node }
if ($PromptNode) { $resolveArgs.Prompt = $true }

Write-Host "Resolving Slurm GPU node..."
$fqdn = & "$scriptDir\slurm_resolve_node.ps1" @resolveArgs
if (-not $fqdn) { throw "Failed to resolve GPU node." }
if ($fqdn -is [array]) { $fqdn = ($fqdn | Where-Object { $_ } | Select-Object -Last 1) }
$fqdn = "$fqdn".Trim()
$nodeShort = ($fqdn -split "\.")[0]
Write-Host "Using GPU node: $fqdn  (ssh $GpuHostAlias)"

if (-not $SkipSync) {
    & "$scriptDir\slurm_sync.ps1" -HostAlias "bgu-slurm" -RemoteRepo $RemoteRepo
}

if (-not $LogDir) {
    $LogDir = Join-Path $repo "runs\slurm_logs"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $LogDir "run_${stamp}_${nodeShort}.log"

$extra = ($ExtraArgs -join " ")

$remoteScript = @"
#!/usr/bin/env bash
set -euo pipefail
cd '$RemoteRepo'
echo "=== SPECTRA slurm run ==="
echo "host=`$(hostname)"
echo "date=`$(date -Iseconds)"
echo "repo=$RemoteRepo"
echo "extra_args=$extra"
bash scripts/check_gpu.sh
bash scripts/run_agent.sh $extra
echo "=== SPECTRA slurm run finished OK ==="
"@
$remoteScript = $remoteScript -replace "`r`n", "`n" -replace "`r", "`n"

$tmpLocal = Join-Path $env:TEMP "spectra_slurm_run.sh"
$tmpRemote = "/tmp/spectra_slurm_run_$env:USERNAME.sh"
[IO.File]::WriteAllText($tmpLocal, $remoteScript, (New-Object System.Text.UTF8Encoding $false))

Write-Host "Running on $GpuHostAlias ($fqdn) ..."
Write-Host "Local log: $logFile"
"=== local wrapper start $(Get-Date -Format o) node=$fqdn ===" | Set-Content -Path $logFile -Encoding utf8

scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new $tmpLocal "${GpuHostAlias}:$tmpRemote"
$exitCode = 0
try {
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new $GpuHostAlias "bash $tmpRemote; ec=`$?; rm -f $tmpRemote; exit `$ec" 2>&1 |
        Tee-Object -FilePath $logFile -Append
    $exitCode = $LASTEXITCODE
} catch {
    $_ | Tee-Object -FilePath $logFile -Append | Write-Host
    $exitCode = 1
} finally {
    Remove-Item -Force $tmpLocal -ErrorAction SilentlyContinue
    "=== local wrapper end $(Get-Date -Format o) exit=$exitCode ===" | Add-Content -Path $logFile -Encoding utf8
}

Write-Host ""
Write-Host "Run log saved: $logFile"
Write-Host "In Cursor: @runs/slurm_logs/$((Split-Path -Leaf $logFile))  (or open the file) for debugging / next steps."
if ($exitCode -ne 0) {
    throw "Remote run failed with exit code $exitCode (see log)."
}
