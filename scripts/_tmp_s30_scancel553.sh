#!/usr/bin/env bash
scancel 21729553 && echo "scancelled 21729553 (pre-registered: R56 rows printed; VGG-16 at step 1 cannot finish by the 08:21 wall)"
sleep 45
squeue -u paretsky -h -o "%.9i %.24j %.2t %.10M %.5Q %R" | sort -k3,3r -k5,5nr | head -8