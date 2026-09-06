# Push working-tree files to the leap overlay without committing.
# Usage: .\scripts\push_overlay.ps1 src\utils.py scripts\submit.sh
# scp is used directly: rexec.ps1 passes its payload on the command line, which
# overflows the Windows command-length limit for source files.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true, Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Paths,
    [string]$SshHost = "bgu-slurm",
    [string]$RemoteRoot = "/home/paretsky/SPECTRA-CompressionAgent"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

$dirs = @{}
foreach ($rel in $Paths) {
    $full = Join-Path $repo $rel
    if (-not (Test-Path -LiteralPath $full)) { throw "missing: $full" }
    $dirs[(Split-Path -Parent ($rel -replace '\\', '/')) -replace '\\', '/'] = $true
}
$mk = ($dirs.Keys | Where-Object { $_ } | ForEach-Object { "'$RemoteRoot/$_'" }) -join " "
if ($mk) {
    ssh -o BatchMode=yes $SshHost "mkdir -p $mk"
    if ($LASTEXITCODE -ne 0) { throw "mkdir failed" }
}

foreach ($rel in $Paths) {
    $full = Join-Path $repo $rel
    $posix = ($rel -replace '\\', '/')
    scp -q -o BatchMode=yes $full "${SshHost}:$RemoteRoot/$posix"
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $posix" }
    Write-Output "pushed $posix"
}
