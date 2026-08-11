<#
.SYNOPSIS
    Run a bash script on the BGU SLURM login node without quoting or line-ending problems.

.DESCRIPTION
    PowerShell -> ssh -> bash passes a string through three different quoting regimes, and
    Windows line endings make bash fail with `$'\r': command not found`. This helper removes
    both classes of failure by normalising to LF and transporting the script as base64, which
    contains no characters that any of the three layers treat specially.

.EXAMPLE
    .\scripts\rexec.ps1 -File scripts/env_report.sh
    .\scripts\rexec.ps1 -Command 'squeue -u $USER; sacct -n | tail -5'
    .\scripts\rexec.ps1 -File scripts/job_status.sh -ArgumentList 12345
#>
param(
    [string]$File,
    [string]$Command,
    [string[]]$ArgumentList = @(),
    [string]$SshHost = "bgu-slurm",
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

if ($File) {
    if (-not (Test-Path $File)) { throw "script not found: $File" }
    $body = Get-Content $File -Raw
} elseif ($Command) {
    $body = $Command
} else {
    throw "supply -File or -Command"
}

# bash rejects CR; normalise regardless of how the file was authored on Windows
$body = $body -replace "`r`n", "`n" -replace "`r", "`n"

# Positional arguments are appended as a `set --` prologue so the script sees $1, $2, ...
if ($ArgumentList.Count -gt 0) {
    $quoted = ($ArgumentList | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join " "
    $body = "set -- $quoted`n" + $body
}

$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($body))

if (-not $Quiet) { Write-Host "-> $SshHost ($($body.Length) bytes)" -ForegroundColor DarkGray }

# `base64 -d` is present on the login and compute nodes; bash reads the decoded script itself
ssh -o BatchMode=yes $SshHost "echo $encoded | base64 -d | bash -s" 2>&1
exit $LASTEXITCODE
