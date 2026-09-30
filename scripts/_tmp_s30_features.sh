#!/usr/bin/env bash
for n in ise-6000-05 ise-4090-10 cs-3090-04 cs-1080-02 cs-2080-01 cs-pheno-06; do
  echo "$n $(scontrol show node $n 2>/dev/null | grep -oE 'AvailableFeatures=[^ ]*|Weight=[^ ]*' | tr '\n' ' ')"
done
scontrol show job 21729558 2>/dev/null | grep -oE 'TresPerNode=[^ ]*|TresPerJob=[^ ]*|Features=[^ ]*|Gres=[^ ]*' | tr '\n' ' '; echo
sinfo --version