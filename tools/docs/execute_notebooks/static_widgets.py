"""
Replace the Jupyter widgets saved in the documentation notebooks with static text.

The only widgets in the documentation are tqdm progress bars. As widgets, they
need the widget manager of mkdocs-jupyter (require.js plus a script loaded
from unpkg.com at runtime) to be displayed, and bars executed without saved
widget state are shown frozen at 0%. This script turns every progress bar with
saved state into the line tqdm prints in a terminal, with its final values:

    100%|██████████| 2/2 [00:00<00:00, 188.42it/s]

Progress bars without saved state are removed, since their only content is the
initial 0% line. The widget state stored in the notebook metadata is removed.
Nothing else changes: the files are rewritten with their original formatting.

tools/docs/execute_notebooks/execute_notebooks.py applies it after executing each
notebook. A notebook run and saved by hand (Jupyter, VS Code) gets its widgets
back: the MkDocs hook tools/docs/hooks/notebook_widgets.py then lists it as a
warning on every docs build, and this script fixes it.

Usage
-----
    python tools/docs/execute_notebooks/static_widgets.py                  # all docs/ notebooks
    python tools/docs/execute_notebooks/static_widgets.py docs/faq/x.ipynb # specific notebooks
    python tools/docs/execute_notebooks/static_widgets.py --check          # exit 1 if any has widgets
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIDGET_VIEW = "application/vnd.jupyter.widget-view+json"
WIDGET_STATE = "application/vnd.jupyter.widget-state+json"
BAR_WIDTH = 10


def _html_text(value: str | None) -> str:
    """Plain text of the value of an HTML widget, as tqdm writes it."""
    return html.unescape(value or "").replace(" ", " ").replace("\xa0", " ")


def _progress_bar_text(model_id: str, state: dict) -> str | None:
    """
    Text of a tqdm progress bar (HBox with HTML, FloatProgress and HTML children)
    from the saved widget state, or None if the state is missing or different.
    """
    box = state.get(model_id)
    if box is None or box.get("model_name") != "HBoxModel":
        return None
    children = [
        state.get(child.replace("IPY_MODEL_", ""), {})
        for child in box["state"].get("children", [])
    ]
    names = [child.get("model_name") for child in children]
    if names != ["HTMLModel", "FloatProgressModel", "HTMLModel"]:
        return None

    prefix = _html_text(children[0]["state"].get("value"))
    progress = children[1]["state"]
    suffix = _html_text(children[2]["state"].get("value"))
    value, total = progress.get("value") or 0, progress.get("max")
    if total:
        filled = round(BAR_WIDTH * min(value / total, 1))
        bar = "█" * filled + " " * (BAR_WIDTH - filled)
    else:
        bar = " " * BAR_WIDTH

    return f"{prefix}|{bar}|{suffix}"


def convert_notebook(path: Path) -> tuple[int, int]:
    """
    Replace the progress bar widgets of a notebook with static text.

    Returns the number of progress bars converted to text and removed. The file
    is only rewritten when something changes.
    """
    raw = path.read_text(encoding="utf-8")
    notebook = json.loads(raw)
    metadata = notebook.get("metadata", {})
    state = metadata.get("widgets", {}).get(WIDGET_STATE, {}).get("state", {})

    converted = removed = 0
    for cell in notebook.get("cells", []):
        outputs = cell.get("outputs")
        if not outputs:
            continue
        kept = []
        for output in outputs:
            view = output.get("data", {}).get(WIDGET_VIEW)
            if view is None:
                kept.append(output)
                continue
            text = _progress_bar_text(view.get("model_id", ""), state)
            if text is None:
                removed += 1
                continue
            output["data"] = {"text/plain": [text]}
            kept.append(output)
            converted += 1
        cell["outputs"] = kept

    changed = converted or removed or "widgets" in metadata
    if changed:
        metadata.pop("widgets", None)
        trailing_newline = "\n" if raw.endswith("\n") else ""
        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + trailing_newline,
            encoding="utf-8",
        )

    return converted, removed


def has_widgets(path: Path) -> bool:
    """Whether a notebook contains widget outputs or saved widget state."""
    text = path.read_text(encoding="utf-8")
    return WIDGET_VIEW in text or WIDGET_STATE in text


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("notebooks", nargs="*", type=Path, help="Notebooks to process.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only list the notebooks with widgets (exit code 1 if any), do not modify.",
    )
    args = parser.parse_args()

    notebooks = args.notebooks or sorted((REPO_ROOT / "docs").rglob("*.ipynb"))
    if args.check:
        found = [path for path in notebooks if has_widgets(path)]
        for path in found:
            print(f"{path}: contains Jupyter widgets")
        if found:
            raise SystemExit(
                f"{len(found)} notebook(s) with widgets. Fix them with: "
                "python tools/docs/execute_notebooks/static_widgets.py"
            )
        print("No notebook contains Jupyter widgets.")
        return

    total_converted = total_removed = 0
    for path in notebooks:
        converted, removed = convert_notebook(path)
        if converted or removed:
            print(f"{path}: {converted} converted, {removed} removed")
        total_converted += converted
        total_removed += removed
    print(
        f"Progress bars converted to text: {total_converted}, removed: {total_removed}"
    )


if __name__ == "__main__":
    main()
