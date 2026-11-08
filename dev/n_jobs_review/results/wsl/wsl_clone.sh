#!/usr/bin/env bash
# Clone the Windows repo into the Linux filesystem at 0.27.x (1a3eadc51) and
# add the study scripts from the local branch docs/n-jobs-review.
set -eu
SRC=/mnt/c/Users/Joaquin/Documents/GitHub/skforecast
DST="$HOME/github/skforecast_njobs_wsl"
mkdir -p "$HOME/github"
if [ ! -d "$DST" ]; then
  git clone -q --no-checkout "$SRC" "$DST"
fi
cd "$DST"
git fetch -q "$SRC" docs/n-jobs-review:docs/n-jobs-review
git checkout -q --detach 1a3eadc51
git checkout docs/n-jobs-review -- dev/n_jobs_review
git reset -q
git log -1 --oneline
ls dev/n_jobs_review
P="$HOME/miniconda3/envs/skforecast_27_py13_wsl/bin/python"
cd "$DST"
PYTHONPATH="$DST" "$P" -c "import skforecast, joblib, sys; print(skforecast.__version__, skforecast.__file__); print('cpu_count', joblib.cpu_count(), joblib.cpu_count(only_physical_cores=True)); import importlib.util as u; print('torch', u.find_spec('torch'))"
