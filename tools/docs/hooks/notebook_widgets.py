"""
MkDocs hook that warns about Jupyter widgets saved in the documentation notebooks.

The documentation does not load the Jupyter widget manager (mkdocs-jupyter runs
with `include_requirejs: False`), so a widget saved in a notebook is not
displayed: a tqdm progress bar shows up empty, or frozen at 0% when the widget
state was not saved. `tools/docs/execute_notebooks/execute_notebooks.py` replaces the
progress bars with static text after executing a notebook, but a notebook run
and saved by hand (Jupyter, VS Code) keeps them. This hook lists those notebooks
as warnings on every build, with the command that fixes them. It does not
modify any file.

Registered in `mkdocs.yml` as:

    hooks:
      - tools/docs/hooks/notebook_widgets.py
"""

import logging
from pathlib import Path

logger = logging.getLogger("mkdocs.hooks.notebook_widgets")

WIDGET_MIME_TYPES = (
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.jupyter.widget-state+json",
)


def on_pre_build(config):
    docs_dir = Path(config["docs_dir"])
    for path in sorted(docs_dir.rglob("*.ipynb")):
        text = path.read_text(encoding="utf-8")
        if any(mime in text for mime in WIDGET_MIME_TYPES):
            logger.warning(
                "Notebook '%s' contains Jupyter widgets (tqdm progress bars), which "
                "the documentation does not display. Convert them to text with: "
                "python tools/docs/execute_notebooks/static_widgets.py %s",
                path.relative_to(docs_dir),
                path.relative_to(docs_dir.parent),
            )
