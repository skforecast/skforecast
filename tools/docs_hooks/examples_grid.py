"""
MkDocs hook that renders the Examples pages from a single YAML source.

The three pages under `docs/examples/` (English, Spanish, Chinese) each contain a
single `<!-- EXAMPLES-GRID -->` marker. This hook replaces that marker with a
language switcher, a filter bar and one Material card grid per section, all built
from `examples.yml` sitting next to this file.

Registered in `mkdocs.yml` as:

    hooks:
      - tools/docs_hooks/examples_grid.py

Notes for maintainers:

- The YAML is read in `on_config`, so editing it during `mkdocs serve` triggers a
  rebuild with the new content (`watch: [tools/docs_hooks]` in `mkdocs.yml` makes
  the server notice the change). Editing *this file* has no effect until the
  server is restarted, because MkDocs caches hook modules for the life of the
  process.
- Rendering happens in `on_page_markdown`, before `page.render()`, so the
  generated `##` headings reach `page.toc` (needed by the `toc.integrate`
  feature) and the language switcher's `.md` links get rewritten and validated.
- Any problem in the YAML raises `PluginError` from `on_config`, which aborts the
  build with a single clean message before any page is rendered.
"""

import html
import logging
from pathlib import Path

import yaml
from mkdocs.exceptions import PluginError

logger = logging.getLogger("mkdocs.hooks.examples_grid")

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "examples.yml"
MARKER = "<!-- EXAMPLES-GRID -->"

REQUIRED_UI_KEYS = (
    "search_label",
    "search_placeholder",
    "level",
    "topic",
    "clear",
    "count",
    "count_all",
    "no_results",
)

# Populated by on_config.
DATA = None


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def _load_icon_index():
    """
    Return the set of icon shortcodes bundled with mkdocs-material.

    Used to catch typos: `pymdownx.emoji` renders an unknown shortcode as plain
    text without warning, so `:material-chart-lien:` would otherwise ship to the
    live site as literal characters.
    """
    try:
        from material.extensions.emoji import twemoji

        return set(twemoji({}, None)["emoji"])
    except Exception as exc:  # defensive: depends on theme internals
        logger.warning(
            "examples_grid: could not load the mkdocs-material icon index (%s). "
            "Icon shortcodes will not be validated.",
            exc,
        )
        return None


def _require(condition, message):
    if not condition:
        raise PluginError(f"tools/docs_hooks/examples.yml: {message}")


def _validate(data):
    """Validate the YAML structure, raising PluginError on the first problem."""
    _require(isinstance(data, dict), "file is empty or not a mapping")
    for key in ("vocabulary", "languages", "ui", "sections", "examples"):
        _require(key in data, f"missing top-level key '{key}'")

    levels = data["vocabulary"].get("levels") or []
    topics = data["vocabulary"].get("topics") or []
    _require(levels, "vocabulary.levels is empty")
    _require(topics, "vocabulary.topics is empty")
    overlap = set(levels) & set(topics)
    _require(
        not overlap,
        f"vocabulary.levels and vocabulary.topics overlap: {sorted(overlap)}",
    )
    known_tags = set(levels) | set(topics)

    languages = data["languages"]
    for code, cfg in languages.items():
        _require(cfg.get("page"), f"languages.{code} is missing 'page'")
        _require(cfg.get("label"), f"languages.{code} is missing 'label'")

    for code in languages:
        _require(code in data["ui"], f"ui.{code} is missing")
        ui = data["ui"][code]
        for key in REQUIRED_UI_KEYS:
            _require(ui.get(key), f"ui.{code} is missing '{key}'")
        labels = ui.get("labels") or {}
        missing = sorted(known_tags - set(labels))
        _require(not missing, f"ui.{code}.labels is missing entries for {missing}")

    section_ids = []
    for section in data["sections"]:
        sid = section.get("id")
        _require(sid, "a section is missing 'id'")
        _require(sid not in section_ids, f"duplicate section id '{sid}'")
        section_ids.append(sid)
        for code in languages:
            block = section.get(code) or {}
            _require(block.get("title"), f"section '{sid}' is missing {code}.title")
            _require(block.get("blurb"), f"section '{sid}' is missing {code}.blurb")

    icon_index = _load_icon_index()
    seen_ids = set()
    for example in data["examples"]:
        eid = example.get("id")
        _require(eid, "an example is missing 'id'")
        _require(eid not in seen_ids, f"duplicate example id '{eid}'")
        seen_ids.add(eid)

        icon = example.get("icon")
        _require(icon, f"example '{eid}' is missing 'icon'")
        if icon_index is not None:
            _require(
                f":{icon}:" in icon_index,
                f"example '{eid}' uses unknown icon shortcode ':{icon}:'",
            )

        _require(
            example.get("section") in section_ids,
            f"example '{eid}' has unknown section '{example.get('section')}'",
        )

        tags = example.get("tags") or []
        unknown = sorted(set(tags) - known_tags)
        _require(
            not unknown,
            f"example '{eid}' has tags outside the vocabulary: {unknown}",
        )
        n_levels = len([tag for tag in tags if tag in levels])
        _require(
            n_levels == 1,
            f"example '{eid}' must have exactly one level tag, found {n_levels}",
        )
        _require(
            any(tag in topics for tag in tags),
            f"example '{eid}' must have at least one topic tag",
        )

        translated = [code for code in languages if example.get(code)]
        _require(translated, f"example '{eid}' has no language blocks")
        for code in translated:
            block = example[code]
            for key in ("title", "summary", "url"):
                _require(block.get(key), f"example '{eid}' is missing {code}.{key}")
            title = block["title"]
            _require(
                "[" not in title and "]" not in title,
                f"example '{eid}' {code}.title contains square brackets, "
                "which would break the generated Markdown link",
            )

    return data


def on_config(config):
    """Read and validate examples.yml once per build (and per serve rebuild)."""
    global DATA

    try:
        raw = yaml.safe_load(DATA_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PluginError(f"examples_grid: {DATA_FILE} not found") from exc
    except yaml.YAMLError as exc:
        raise PluginError(
            f"examples_grid: {DATA_FILE} is not valid YAML: {exc}"
        ) from exc

    DATA = _validate(raw)
    return config


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _tags_in_order(values, present):
    """Keep `values` in vocabulary order, dropping anything absent from the page."""
    return [value for value in values if value in present]


def _render_language_switcher(lang):
    """A row of links to the same page in the other languages."""
    parts = []
    for code, cfg in DATA["languages"].items():
        classes = ".ex-lang"
        if code == lang:
            classes += " .ex-lang--active aria-current=page"
        parts.append(f"[{cfg['label']}]({cfg['page']}){{ {classes} }}")
    return "\n".join(parts) + "\n{ .ex-langs }"


def _render_chip(group, value, label):
    return (
        f'    <button class="ex-chip" type="button" data-ex-group="{group}" '
        f'data-ex-value="{html.escape(value, quote=True)}" aria-pressed="false">'
        f"{html.escape(label)}</button>"
    )


def _render_filter_bar(lang, examples):
    """
    The search box, the two chip groups and the result count.

    Rendered with `hidden` and unhidden by examples-filter.js, so that with
    JavaScript disabled the page is simply a full, unfiltered card grid.

    `data-search-exclude` keeps the interface strings out of the site search
    index. The wrapper is a `<section>` rather than a `<div>` on purpose:
    Material's search parser tracks excluded elements by tag name only, so a
    nested `<div>` closing would cancel the exclusion of an outer `<div>` and
    leak every chip label into the index.
    """
    ui = DATA["ui"][lang]
    labels = ui["labels"]
    present = {tag for example in examples for tag in example.get("tags", [])}
    levels = _tags_in_order(DATA["vocabulary"]["levels"], present)
    topics = _tags_in_order(DATA["vocabulary"]["topics"], present)
    total = len(examples)

    search_label = html.escape(ui["search_label"])
    placeholder = html.escape(ui["search_placeholder"], quote=True)

    # Layout (see extra.css): the search box spans the full width, each chip group
    # is a label column plus a chip column, and the count/reset footer closes.
    lines = [
        '<section class="ex-filter" id="ex-filter" hidden data-search-exclude>',
        '  <div class="ex-filter__search">',
        f'    <label class="ex-visually-hidden" for="ex-q">{search_label}</label>',
        '    <input class="ex-filter__input" id="ex-q" type="search"',
        '           autocomplete="off" spellcheck="false"',
        f'           placeholder="{placeholder}">',
        "  </div>",
    ]

    for group, values in (("level", levels), ("topic", topics)):
        if not values:
            continue
        group_label = html.escape(ui[group])
        lines.append(
            f'  <div class="ex-chips" role="group" aria-label="{group_label}">'
        )
        lines.append(f'    <span class="ex-chips__label">{group_label}</span>')
        lines.append('    <div class="ex-chips__list">')
        lines.extend(_render_chip(group, value, labels[value]) for value in values)
        lines.append("    </div>")
        lines.append("  </div>")

    count_all = ui["count_all"].format(total=total)
    lines.extend(
        [
            '  <div class="ex-filter__footer">',
            '    <p class="ex-filter__count" data-ex-count aria-live="polite"',
            '       aria-atomic="true"',
            f'       data-ex-count-all="{html.escape(ui["count_all"], quote=True)}"',
            f'       data-ex-count-some="{html.escape(ui["count"], quote=True)}">'
            f"{html.escape(count_all)}</p>",
            '    <button class="ex-chip ex-chip--reset" type="button" data-ex-reset'
            f' hidden>{html.escape(ui["clear"])}</button>',
            "  </div>",
            "</section>",
        ]
    )
    return "\n".join(lines)


def _render_card(lang, example):
    """
    One Material card.

    The three nesting levels are all required: Material styles
    `.grid.cards > ul > li`, so the `<ul>` cannot be the grid container itself,
    and `md_in_html` needs `markdown` on the wrapper and `markdown="block"` on
    the item (a bare `markdown` silently switches to inline mode and collapses
    the card body into a single paragraph).

    The card header is one paragraph holding the icon (styled as a badge by
    extra.css) and the bold title link. The level tag is always listed first
    and carries a modifier class so the stylesheet can give it a colored dot.
    """
    block = example[lang]
    labels = DATA["ui"][lang]["labels"]
    tags = example.get("tags", [])
    level = next(tag for tag in tags if tag in DATA["vocabulary"]["levels"])
    topics = [tag for tag in DATA["vocabulary"]["topics"] if tag in tags]
    chips = [
        f'<span class="ex-tag ex-tag--level ex-tag--{level}">'
        f"{html.escape(labels[level])}</span>"
    ]
    chips.extend(
        f'<span class="ex-tag">{html.escape(labels[tag])}</span>' for tag in topics
    )
    tag_attr = html.escape(" ".join(tags), quote=True)
    title_link = f'**[{block["title"]}]({block["url"]})**'

    return "\n".join(
        [
            f'<li class="ex-card" data-ex-tags="{tag_attr}" markdown="block">',
            f':{example["icon"]}:{{ .ex-card__icon }} {title_link}',
            "{ .ex-card__title }",
            "",
            block["summary"],
            "{ .ex-card__summary }",
            "",
            f'<p class="ex-card__tags">{"".join(chips)}</p>',
            "</li>",
        ]
    )


def _render_section(lang, section, examples):
    sid = section["id"]
    block = section[lang]
    cards = "\n".join(_render_card(lang, example) for example in examples)

    return "\n".join(
        [
            f'## {block["title"]} {{ #{sid} data-ex-section="{sid}" }}',
            "",
            block["blurb"],
            f'{{ .ex-section-intro data-ex-section="{sid}" }}',
            "",
            f'<div class="grid cards ex-grid" markdown data-ex-section="{sid}">',
            "<ul markdown>",
            cards,
            "</ul>",
            "</div>",
        ]
    )


def _render_page(lang):
    ui = DATA["ui"][lang]
    examples = [example for example in DATA["examples"] if example.get(lang)]

    parts = [
        _render_language_switcher(lang),
        _render_filter_bar(lang, examples),
    ]

    if ui.get("fallback_note"):
        # Body rather than title, so the note can contain Markdown links, and an
        # empty title so no untranslated "Info" heading appears.
        parts.append(f'!!! info ""\n\n    {ui["fallback_note"]}')

    for section in DATA["sections"]:
        in_section = [
            example for example in examples if example["section"] == section["id"]
        ]
        if in_section:
            parts.append(_render_section(lang, section, in_section))

    # data-search-exclude keeps the interface strings out of the search index.
    no_results = html.escape(ui["no_results"])
    parts.append(
        f'<p class="ex-noresults" hidden data-search-exclude>{no_results}\n'
        '  <button class="ex-chip" type="button" data-ex-reset>'
        f'{html.escape(ui["clear"])}</button></p>'
    )

    return "\n\n".join(parts)


def on_page_markdown(markdown, *, page, config, files):
    """Replace the marker on the three examples pages."""
    src_path = page.file.src_uri
    for lang, cfg in DATA["languages"].items():
        if src_path != f"examples/{cfg['page']}":
            continue
        if MARKER not in markdown:
            raise PluginError(
                f"examples_grid: {src_path} does not contain the {MARKER} marker"
            )
        return markdown.replace(MARKER, _render_page(lang))
    return markdown
