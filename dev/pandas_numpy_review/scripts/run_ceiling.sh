#!/usr/bin/env bash
D=$(dirname "$0"); export OMP_NUM_THREADS=1
P3=~/.venvs/skforecast/bin/python; P2=~/.venvs/skforecast-p2/bin/python
for est in ridge lgbm; do for n in 100 600; do for ex in none float mixed; do
  $P2 $D/ceiling_bt_ms.py $n $ex $est 50 24
  NO_PYARROW=1 $P3 $D/ceiling_bt_ms.py $n $ex $est 50 24
  $P3 $D/ceiling_bt_ms.py $n $ex $est 50 24
done; done; done
