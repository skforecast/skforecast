---
paths:
  - "docs/**"
  - "mkdocs.yml"
  - "README.md"
---

# Documentation

## Executing notebooks

Notebooks are committed with their outputs. Re-execute them with:

```bash
python tools/docs/execute_notebooks/execute_notebooks.py [subdir_or_notebook]
```

It runs papermill and writes warning logs to `tools/docs/execute_notebooks/logs/`. Notebooks listed in `SLOW_NOTEBOOKS` inside that script are skipped unless `--include-slow` is passed or the notebook is given explicitly. Executing the whole `docs/` tree is slow, so pass the specific subdirectory or notebook that changed. It exits with 1 if a notebook fails or the run is interrupted.

`--check` executes nothing: it scans the saved outputs of every notebook in scope (excluded and slow ones included) in under a second, and exits with 1 if any has error outputs or unexecuted code cells (which would be published as missing or broken outputs), or if a `*_temp_exec.ipynb` file of a killed run is left under `docs/`.

The kernel runs from the notebook's folder, where the repo is not on the path: if the environment has a regular (non-editable) install of skforecast, the notebook silently runs that version instead of the repo code. The runner reports it in its header (`Kernel skf`, with a warning when it is not the repo code); to check it by hand, run `cd docs/user_guides && python -c "import skforecast; print(skforecast.__file__)"`. If it points to `site-packages`, run `pip install -e .` or prefix the command with `PYTHONPATH=<repo root>`.

## Keep user guides simple

- Write example code for the result the notebook actually produces (outputs are committed). If RFECV keeps `roll_mean_24` and `roll_mean_48`, write `RollingFeatures(stats=['mean', 'mean'], window_sizes=[24, 48])`, not a parser that handles any name.
- Avoid defensive branches, helper loops and bookkeeping (timers, counters, dicts of candidates) unless they are the point of the example.
- Say each thing once; link to the API reference or another section instead of repeating parameter descriptions. Prefer a short sentence to an admonition, and a few takeaways and links to long lists.
- Overview and introduction pages are even lighter: explain the idea at the level of the concept, name the most flexible option and link to the detailed guide. Do not add capability matrices or every exception.

## Foundation models before RNN

`ForecasterFoundation` carries more weight than `ForecasterRnn` wherever both appear (docs, READMEs, skills, `llms-base.txt`):

- Place Foundation before RNN in tables, section order, lists and diagrams; give it more depth and keep RNN brief. Do not propose dedicated effort for RNN unless asked.
- Present foundation models as a real forecasting option, not a mere baseline: strong zero-shot accuracy (top of GIFT-Eval and fev-bench, good with short or new series, no training), with honest costs (more compute, usually a GPU; a model trained with domain features can still win on a given dataset).
