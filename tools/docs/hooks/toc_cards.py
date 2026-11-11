"""
MkDocs hook that renders the table of contents pages as card grids from `nav`.

Each page listed in `toc_cards.yml`, sitting next to this file, contains a
single `<!-- TOC-CARDS -->` marker. This hook replaces it with one card per
section of the entry of `nav` in `mkdocs.yml` that the YAML names for the page,
so the page cannot drift from the sidebar: adding a page to `nav` adds it to
the cards. The YAML holds the texts `nav` cannot say: one sentence per section,
optionally a longer title for a page and links to pages outside the entry.

Registered in `mkdocs.yml` as:

    hooks:
      - tools/docs/hooks/toc_cards.py

Notes for maintainers:

- The YAML is read in `on_config`, so editing it during `mkdocs serve` triggers a
  rebuild with the new content (`watch: [tools/docs/hooks]` in `mkdocs.yml`).
  Editing *this file* has no effect until the server is restarted.
- Rendering happens in `on_page_markdown`, so the links are rewritten and
  validated by MkDocs. Each card title is a `##` heading with its own anchor,
  but `on_page_content` empties `page.toc`: with `toc.integrate` the sidebar
  would list the sections twice, as headings of the page and as navigation
  (`hide: toc` in the page does not remove the integrated list).
- A section of `nav` without a sentence, or an entry of the YAML that no longer
  matches `nav`, is reported as a warning: the page is still built, and
  `mkdocs build --strict` fails.
- The cards are styled by the "Table of Contents Cards" block of
  `docs/stylesheets/extra.css`.
"""

import logging
import posixpath
from pathlib import Path

import yaml
from mkdocs.exceptions import PluginError
from mkdocs.structure.toc import get_toc

logger = logging.getLogger("mkdocs.hooks.toc_cards")

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "toc_cards.yml"
MARKER = "<!-- TOC-CARDS -->"

# Populated by on_config: {page path: [(section title, sentence, [(title, url)])]}.
PAGES = {}


def _nav_sections(nav, nav_entry, toc_page):
    """
    Return the sections of the `nav_entry` entry of `nav` as a list of
    (section title, [(page title, path)]). A page that is not inside a section
    is a section with a single page. The table of contents page is skipped.
    """
    entries = next(
        (item[nav_entry] for item in nav or [] if nav_entry in item), None
    )
    if not isinstance(entries, list):
        raise PluginError(
            f"toc_cards: mkdocs.yml has no '{nav_entry}' section in nav"
        )

    sections = []
    for entry in entries:
        (title, content), = entry.items()
        if content == toc_page:
            continue
        if isinstance(content, str):
            sections.append((title, [(title, content)]))
            continue
        pages = []
        for page in content:
            (page_title, path), = page.items()
            if not isinstance(path, str):
                raise PluginError(
                    f"toc_cards: '{title}' > '{page_title}' is a nested "
                    f"section, which the table of contents does not support"
                )
            pages.append((page_title, path))
        sections.append((title, pages))

    return sections


def _page_sections(toc_page, data, nav):
    """
    Return the sections of the cards of `toc_page` as a list of
    (section title, sentence, [(title, url)]), warning about the entries of its
    YAML `data` that do not match `nav`.
    """
    nav_entry = data.get("nav")
    if not nav_entry:
        raise PluginError(f"toc_cards: {toc_page} has no 'nav' in {DATA_FILE.name}")

    sentences = data.get("sections") or {}
    titles = data.get("titles") or {}
    extra_links = data.get("extra_links") or {}
    nav_sections = _nav_sections(nav, nav_entry, toc_page)
    nav_titles = {title for title, _ in nav_sections}
    nav_paths = {path for _, pages in nav_sections for _, path in pages}

    for title in nav_titles - set(sentences):
        logger.warning(
            "toc_cards: section '%s' of '%s' in nav has no sentence in %s",
            title,
            nav_entry,
            DATA_FILE.name,
        )
    for key, names in (("a sentence", sentences), ("extra links", extra_links)):
        for title in set(names) - nav_titles:
            logger.warning(
                "toc_cards: %s has %s for '%s', which is not a section of '%s' "
                "in nav",
                DATA_FILE.name,
                key,
                title,
                nav_entry,
            )
    for path in set(titles) - nav_paths:
        logger.warning(
            "toc_cards: %s has a title for '%s', which is not a page of '%s' in "
            "nav",
            DATA_FILE.name,
            path,
            nav_entry,
        )

    sections = []
    for title, pages in nav_sections:
        links = [(titles.get(path, page_title), path) for page_title, path in pages]
        for link in extra_links.get(title) or []:
            if not isinstance(link, dict) or not {"title", "url"} <= set(link):
                raise PluginError(
                    f"toc_cards: each extra link of '{title}' in "
                    f"{DATA_FILE.name} needs a 'title' and a 'url'"
                )
            links.append((link["title"], link["url"]))
        sections.append((title, sentences.get(title), links))

    return sections


def on_config(config):
    global PAGES

    try:
        data = yaml.safe_load(DATA_FILE.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise PluginError(f"toc_cards: {DATA_FILE} not found") from exc
    except yaml.YAMLError as exc:
        raise PluginError(
            f"toc_cards: {DATA_FILE} is not valid YAML: {exc}"
        ) from exc

    PAGES = {
        toc_page: _page_sections(toc_page, page_data or {}, config["nav"])
        for toc_page, page_data in data.items()
    }

    return config


def _href(url, toc_page):
    """
    Return the link from `toc_page` to `url`: absolute URLs are kept, paths of
    `docs/` are made relative to the page so MkDocs rewrites and validates them.
    """
    if "://" in url:
        return url

    return posixpath.relpath(url, posixpath.dirname(toc_page))


def _render(toc_page):
    """
    Return the Markdown of the card grid: one list item per section with its
    title, its sentence and its links.
    """
    lines = ['<div class="grid cards toc-grid" markdown>', ""]
    for title, sentence, links in PAGES[toc_page]:
        lines += [f"-   ## {title}", ""]
        if sentence:
            lines += [f"    {sentence}", ""]
        for link_title, url in links:
            lines.append(f"    - [{link_title}]({_href(url, toc_page)})")
        lines.append("")
    lines.append("</div>")

    return "\n".join(lines)


def on_page_markdown(markdown, *, page, config, files):
    toc_page = page.file.src_uri
    if toc_page not in PAGES:
        return markdown
    if MARKER not in markdown:
        raise PluginError(f"toc_cards: {toc_page} does not contain the {MARKER} marker")

    return markdown.replace(MARKER, _render(toc_page))


def on_page_content(html, *, page, config, files):
    if page.file.src_uri in PAGES:
        page.toc = get_toc([])

    return html
