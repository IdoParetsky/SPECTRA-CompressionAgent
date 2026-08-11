#!/usr/bin/env bash
# Discover how this cluster wants GPUs requested, and what is currently free.
echo "=== sinteractive ==="
command -v sinteractive && sed -n '1,80p' "$(command -v sinteractive)"

echo "=== gres types ==="
sinfo -o "%20N %10c %10m %30G %10t" | sort -u | head -40

echo "=== my jobs ==="
squeue -u "$USER" -o "%.10i %.12P %.20j %.8T %.10M %.6D %R"

echo "=== account limits ==="
sacctmgr -n show assoc where user="$USER" format=Account,Partition,GrpTRES,MaxJobs 2>/dev/null | head
