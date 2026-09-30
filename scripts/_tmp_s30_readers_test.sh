#!/usr/bin/env bash
D=/home/paretsky/scratch_audit/readers_s30; PY=/home/paretsky/.conda/envs/spectra/bin/python
cd $D && $PY -m pytest -q -p no:cacheprovider tests/test_v9c_readouts.py 2>&1 | tail -3
$PY scripts/final_ft_readout.py /home/paretsky/scratch_audit/tree_v9c/runs/job21730499 | grep -E 'size_param|val_best' | cut -c1-175
$PY scripts/final_ft_readout.py /home/paretsky/scratch_audit/tree_v9b/runs/job21729551 | grep -E 'size_param|val_best' | cut -c1-175