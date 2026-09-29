"""
Replace the Jupyter widgets saved in the documentation notebooks with static
progress bars.

The only widgets in the documentation are tqdm progress bars. As widgets, they
need the widget manager of mkdocs-jupyter (require.js plus a script loaded
from unpkg.com at runtime) to be displayed, and bars saved without widget state
(notebooks run in an editor that does not store it, such as VS Code) are shown
frozen at 0%. This script turns every progress bar with saved state into a
static output with its final values:

- text/html: the bar as tqdm draws it in a notebook (label, green bar and
  counters), with inline styles only, so it looks the same in the docs (light
  and dark schemes), in Jupyter and in VS Code, without any JavaScript.
- text/plain: the line tqdm prints in a terminal, as a fallback.

    100%|██████████| 2/2 [00:00<00:00, 188.42it/s]

Progress bars without saved state are removed, since their only content is the
initial 0% line. The widget state stored in the notebook metadata is removed.
Bars already converted to text by an earlier version of this script get their
HTML version from that text. Nothing else changes: the files are rewritten with
their original formatting, and running the script twice changes nothing.

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
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIDGET_VIEW = "application/vnd.jupyter.widget-view+json"
WIDGET_STATE = "application/vnd.jupyter.widget-state+json"
BAR_WIDTH = 10
# Colors of the ipywidgets progress bar: finished, interrupted, running.
BAR_COLORS = {"success": "#388e3c", "danger": "#d32f2f", "info": "#1e88e5"}
# A tqdm line: "[description: ]NN%|bar|counters"
TEXT_BAR = re.compile(
    r"^(?P<label>[^|\n]*?(?P<pct>\d{1,3})%)\|[^|\n]*\|(?P<suffix>[^\n]*)$"
)


def _html_text(value: str | None) -> str:
    """Plain text of the value of an HTML widget, as tqdm writes it."""
    return html.unescape(value or "").replace("\u2007", " ").replace("\xa0", " ")


def _progress_bar_html(label: str, fraction: float, suffix: str, color: str) -> str:
    """
    HTML of a static progress bar, with inline styles only. The track is a
    translucent gray, so it works on light and dark backgrounds.
    """
    width = f"{100 * max(0.0, min(fraction, 1.0)):.1f}".rstrip("0").rstrip(".")
    return (
        '<div style="display:flex;align-items:center;gap:0.6em;margin:0.15em 0">'
        f"<span>{html.escape(label.strip())}</span>"
        '<span style="flex:0 1 18em;height:1em;background:rgba(128,128,128,0.25);'
        'border-radius:2px;overflow:hidden">'
        f'<span style="display:block;width:{width}%;height:100%;background:{color}">'
        "</span></span>"
        f"<span>{html.escape(suffix.strip())}</span>"
        "</div>"
    )


def _html_from_text(text: str) -> str | None:
    """HTML version of a tqdm line converted earlier, or None if it is not one."""
    m = TEXT_BAR.match(text)
    if m is None:
        return None
    fraction = int(m.group("pct")) / 100
    color = BAR_COLORS["success"] if fraction >= 1 else BAR_COLORS["info"]
    return _progress_bar_html(m.group("label"), fraction, m.group("suffix"), color)


def _progress_bar(model_id: str, state: dict) -> dict | None:
    """
    Static output data (text/html and text/plain) of a tqdm progress bar (HBox
    with HTML, FloatProgress and HTML children) from the saved widget state, or
    None if the state is missing or different.
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
    fraction = min(value / total, 1) if total else 0.0
    filled = round(BAR_WIDTH * fraction)
    bar = "█" * filled + " " * (BAR_WIDTH - filled)
    color = BAR_COLORS.get(progress.get("bar_style") or "info", BAR_COLORS["info"])
    if fraction >= 1 and progress.get("bar_style") != "danger":
        color = BAR_COLORS["success"]

    return {
        "text/html": [_progress_bar_html(prefix, fraction, suffix, color)],
        "text/plain": [f"{prefix}|{bar}|{suffix}"],
    }


def convert_notebook(path: Path) -> tuple[int, int]:
    """
    Replace the progress bar widgets of a notebook with static progress bars.

    Returns the number of progress bars converted (including bars already in
    text that get their HTML version) and removed. The file is only rewritten
    when something changes.
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
            data = output.get("data", {})
            view = data.get(WIDGET_VIEW)
            if view is None:
                # Bar converted to text by an earlier version: add its HTML
                if output.get("output_type") == "display_data" and set(data) == {
                    "text/plain"
                }:
                    bar_html = _html_from_text("".join(data["text/plain"]))
                    if bar_html is not None:
                        data["text/html"] = [bar_html]
                        converted += 1
                kept.append(output)
                continue
            bar = _progress_bar(view.get("model_id", ""), state)
            if bar is None:
                removed += 1
                continue
            output["data"] = bar
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
    print(f"Progress bars converted: {total_converted}, removed: {total_removed}")


if __name__ == "__main__":
    main()
