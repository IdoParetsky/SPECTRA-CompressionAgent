#!/usr/bin/env pwsh
# Resolve the current sinteractive GPU node and refresh ~/.ssh/config Host bgu-slurm-gpu.
#
# Default: parse RUNNING jobs for paretsky from the login node (squeue).
# If zero / multiple jobs, or -Prompt: show a Windows input popup.
#
# Usage:
#   .\scripts\slurm_resolve_node.ps1
#   .\scripts\slurm_resolve_node.ps1 -Prompt
#   .\scripts\slurm_resolve_node.ps1 -Node cs-1080-05
#   $n = .\scripts\slurm_resolve_node.ps1 -Quiet
param(
    [string]$LoginHostAlias = "bgu-slurm",
    [string]$GpuHostAlias = "bgu-slurm-gpu",
    [string]$NodeDomain = "auth.ad.bgu.ac.il",
    [string]$SshConfigPath = "$env:USERPROFILE\.ssh\config",
    [string]$Node = "",
    [switch]$Prompt,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$Msg) {
    if (-not $Quiet) { Write-Host $Msg }
}

function ConvertTo-Fqdn([string]$HostShort) {
    $h = $HostShort.Trim()
    if (-not $h) { return $h }
    if ($h.Contains(".")) { return $h }
    return "$h.$NodeDomain"
}

function Get-RunningSlurmNodes {
    $raw = ssh -o BatchMode=yes -o ConnectTimeout=25 $LoginHostAlias `
        "squeue -u `$USER -t R -h -o '%i|%T|%N|%j' 2>/dev/null" 2>$null
    if (-not $raw) { return @() }

    $jobs = @()
    foreach ($line in ($raw -split "`r?`n")) {
        $line = $line.Trim()
        if (-not $line) { continue }
        $parts = $line -split "\|", 4
        if ($parts.Count -lt 3) { continue }
        $jobId = $parts[0]
        $nodelist = $parts[2]
        $name = if ($parts.Count -ge 4) { $parts[3] } else { "" }

        $expanded = ssh -o BatchMode=yes -o ConnectTimeout=25 $LoginHostAlias `
            "scontrol show hostnames '$nodelist'" 2>$null
        $hosts = @()
        if ($expanded) {
            $hosts = @($expanded -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        } else {
            $hosts = @($nodelist)
        }

        foreach ($h in $hosts) {
            $jobs += [pscustomobject]@{
                JobId = $jobId
                Name  = $name
                Node  = $h
                Fqdn  = (ConvertTo-Fqdn $h)
            }
        }
    }
    return $jobs
}

function Show-NodeInputPopup([string]$DefaultValue, [string]$Message) {
    Add-Type -AssemblyName Microsoft.VisualBasic | Out-Null
    $title = "SPECTRA Slurm GPU node"
    return [Microsoft.VisualBasic.Interaction]::InputBox($Message, $title, $DefaultValue)
}

function Update-GpuSshHost([string]$Fqdn) {
    if (-not (Test-Path $SshConfigPath)) {
        throw "SSH config not found: $SshConfigPath"
    }

    $lines = Get-Content -Path $SshConfigPath
    $out = New-Object System.Collections.Generic.List[string]
    $inBlock = $false
    $foundHost = $false
    $wroteHostName = $false

    foreach ($line in $lines) {
        if ($line -match '^\s*Host\s+(\S+)') {
            $thisHost = $Matches[1]
            if ($inBlock -and -not $wroteHostName) {
                $out.Add("    HostName $Fqdn")
                $wroteHostName = $true
            }
            $inBlock = ($thisHost -eq $GpuHostAlias)
            if ($inBlock) {
                $foundHost = $true
                $wroteHostName = $false
            }
            $out.Add($line)
            continue
        }

        if ($inBlock -and ($line -match '^\s*HostName\s+\S+')) {
            $out.Add("    HostName $Fqdn")
            $wroteHostName = $true
            continue
        }

        $out.Add($line)
    }

    if ($inBlock -and -not $wroteHostName) {
        $out.Add("    HostName $Fqdn")
        $wroteHostName = $true
    }

    if (-not $foundHost) {
        throw "Host '$GpuHostAlias' not found in $SshConfigPath - add the block first."
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [IO.File]::WriteAllLines($SshConfigPath, $out.ToArray(), $utf8NoBom)
}

# --- resolve ---
$chosenFqdn = $null
$chosenMeta = $null
$jobs = @()

if ($Node) {
    $chosenFqdn = ConvertTo-Fqdn $Node
    $chosenMeta = "manual -Node"
} else {
    Write-Info "Querying RUNNING Slurm jobs via $LoginHostAlias ..."
    $jobs = @(Get-RunningSlurmNodes)

    if (-not $Prompt -and $jobs.Count -eq 1) {
        $chosenFqdn = $jobs[0].Fqdn
        $chosenMeta = "job $($jobs[0].JobId) ($($jobs[0].Name))"
        Write-Info "Auto-detected node: $($jobs[0].Node)  [$chosenMeta]"
    } elseif (-not $Prompt -and $jobs.Count -gt 1) {
        Write-Info "Multiple RUNNING jobs:"
        for ($i = 0; $i -lt $jobs.Count; $i++) {
            Write-Info ("  [{0}] job {1}  node={2}  name={3}" -f ($i + 1), $jobs[$i].JobId, $jobs[$i].Node, $jobs[$i].Name)
        }
        $Prompt = $true
    } elseif (-not $Prompt -and $jobs.Count -eq 0) {
        Write-Info "No RUNNING Slurm jobs found for this user."
        $Prompt = $true
    }

    if ($Prompt) {
        $default = if ($jobs.Count -ge 1) { $jobs[0].Node } else { "cs-1080-05" }
        if ($jobs.Count -eq 0) {
            $hint = "No active sinteractive job found.`r`nEnter the compute node (short or FQDN), e.g. cs-1080-05"
        } elseif ($jobs.Count -eq 1) {
            $hint = "Confirm or edit the GPU node for job $($jobs[0].JobId):"
        } else {
            $lines = for ($i = 0; $i -lt $jobs.Count; $i++) {
                "[{0}] {1} (job {2})" -f ($i + 1), $jobs[$i].Node, $jobs[$i].JobId
            }
            $hint = "Multiple jobs - enter a node name or index 1..$($jobs.Count):`r`n" + ($lines -join "`r`n")
        }

        $input = Show-NodeInputPopup -DefaultValue $default -Message $hint
        if (-not $input -or -not $input.Trim()) {
            throw "No Slurm node provided (popup cancelled or empty)."
        }
        $input = $input.Trim()

        if ($input -match '^\d+$') {
            $idx = [int]$input - 1
            if ($idx -lt 0 -or $idx -ge $jobs.Count) {
                throw "Invalid job index: $input"
            }
            $chosenFqdn = $jobs[$idx].Fqdn
            $chosenMeta = "popup index $($idx + 1) / job $($jobs[$idx].JobId)"
        } else {
            $chosenFqdn = ConvertTo-Fqdn $input
            $chosenMeta = "popup"
        }
    }
}

if (-not $chosenFqdn) {
    throw "Could not resolve Slurm GPU node."
}

Write-Info "Updating SSH Host $GpuHostAlias -> HostName $chosenFqdn ($chosenMeta)"
Update-GpuSshHost -Fqdn $chosenFqdn
Write-Info "Done. Use: ssh $GpuHostAlias"

# Emit FQDN for callers that capture output
Write-Output $chosenFqdn
