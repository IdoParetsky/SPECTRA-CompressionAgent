#!/usr/bin/env bash
# §150 PASSED (r56-w4 +2.3 pp TEST at equal keep; r20 guard -1.8 mean): Stage-4 train = P + crop+flip.
scontrol release 21737123 && echo "released 21737123 (P+aug)"
scancel 21737095 && echo "cancelled 21737095 (P-only, never started)"
sleep 4
squeue -u paretsky -h -o "%.10i %.22j %.2t %.10M %R" 2>/dev/null | grep -E "area-train|smoke-from|aug-thin-12|ft100-dg-r56"
sprio -u paretsky -h -o "%.10i %.8Y %.6N" 2>/dev/null | sort -k2 -nr | head -4