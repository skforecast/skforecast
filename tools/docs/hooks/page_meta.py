"""
MkDocs hook that sets the search engine title and description of each page.

mkdocs-jupyter ignores front matter in notebooks, and reading it from the
notebook metadata would mean editing every notebook. Instead, the texts live in
`page_meta.yml`, next to this file, keyed by the path of the page relative to
`docs/`. For each page listed there, the hook fills `page.meta`:

- `seo_title`, from `title`: the `<title>` tag, followed by " - <site_name>"
  except on the home page, and the Open Graph title (`docs/overrides/main.html`
  and `home.html`). It is not stored as `page.meta.title`, which Material would
  also show in the header bar: the short title of `nav` in `mkdocs.yml` keeps
  naming the page in the header, the sidebar, the breadcrumbs and the footer.
- `description`: the meta description, also used by the Open Graph and Twitter
  tags of `docs/overrides/main.html`. Without it, a page gets `site_description`.

The hook also writes the canonical URL of every page in lowercase, which the
sitemap reuses. Netlify, which serves skforecast.org, answers any URL with
uppercase letters (`api/ForecasterRecursive.html`) with a 301 redirect to its
lowercase form, and this cannot be disabled. A canonical URL or a sitemap entry
that redirects sends search engines contradictory signals, while the lowercase
URL (`api/forecasterrecursive.html`) is served directly. Files, `nav` and links
keep their names. If the site moves to a host with case-sensitive paths, remove
`_lowercase_canonical_url`, or the canonical URL of those pages would be a 404.
`docs/overrides/main.html` lowercases `og:url` for the same reason.

Registered in `mkdocs.yml` as:

    hooks:
      - tools/docs/hooks/page_meta.py

Every page needs an entry with both texts. The rules of `RULES` are checked at
build time, as warnings (`mkdocs build --strict` fails), and without building
the site, in CI and by hand, with:

    python tools/docs/hooks/page_meta.py

which exits with 1 if a page has no entry, an entry matches no page, or a text
breaks a rule.

Notes for maintainers:

- The YAML is read in `on_config`, so editing it during `mkdocs serve` triggers a
  rebuild with the new texts (`watch: [tools/docs/hooks]` in `mkdocs.yml`).
  Editing *this file* has no effect until the server is restarted.
- Front matter of a Markdown page wins over the YAML, but the check still
  requires the entry, so that all the texts can be reviewed in one place.
"""

import logging
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

logger = logging.getLogger("mkdocs.hooks.page_meta")

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "page_meta.yml"
DOCS_DIR = HERE.parents[2] / "docs"
KEYS = ("title", "description")
# Key of `page.meta` where each text of the YAML is stored
META_KEYS = {"title": "seo_title", "description": "description"}
HOME = "README.md"
# mkdocs-jupyter also takes `.py` files as pages, and `exclude_docs` is applied
# after `on_files`, so the pages are selected by their extension.
PAGE_SUFFIXES = (".md", ".ipynb")
RULES = {
    # The other titles are followed by " - Skforecast Docs" (18 characters),
    # and a search result shows about 60 characters.
    "title_max": 42,
    "home_title_max": 60,
    "description_min": 120,
    "description_max": 160,
}
DASHES = ("\u2013", "\u2014")  # en and em dashes
# Material writes the title and the description without escaping them
# (MkDocs templates are not autoescaped), so these would break the HTML.
UNSAFE = ('"', "<", ">")

# Populated by on_config: {path relative to docs/: {"title": ..., "description": ...}}
PAGE_META = None


def load(path=DATA_FILE):
    """
    Read the YAML and return {page path: {"title": ..., "description": ...}},
    with the texts stripped. Raise ValueError if an entry is malformed.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for page, entry in data.items():
        if not isinstance(entry, dict) or not set(entry) <= set(KEYS):
            raise ValueError(
                f"page_meta: entry '{page}' of {path.name} must be a mapping "
                f"with keys among {list(KEYS)}"
            )
        data[page] = {key: str(value).strip() for key, value in entry.items()}
    return data


def check(data, pages):
    """
    Return the problems of `data` (see `load`) for the documentation `pages`
    (paths relative to `docs/`) as a list of messages.
    """
    problems = []
    for page in sorted(set(pages) - set(data)):
        problems.append(f"'{page}' has no entry in {DATA_FILE.name}")
    for page in sorted(set(data) - set(pages)):
        problems.append(f"'{page}' in {DATA_FILE.name} matches no page")

    counts = {key: Counter(e.get(key) for e in data.values()) for key in KEYS}
    for page, entry in sorted(data.items()):
        for key in KEYS:
            text = entry.get(key, "")
            if not text:
                problems.append(f"'{page}' has no {key}")
                continue
            if counts[key][text] > 1:
                problems.append(f"'{page}' shares its {key} with another page")
            if any(dash in text for dash in DASHES):
                problems.append(f"'{page}' {key} contains an en or em dash")
            if any(char in text for char in UNSAFE):
                problems.append(f"'{page}' {key} contains one of {' '.join(UNSAFE)}")

        title_max = RULES["home_title_max" if page == HOME else "title_max"]
        title = entry.get("title", "")
        if len(title) > title_max:
            problems.append(
                f"'{page}' title has {len(title)} characters (max {title_max})"
            )
        description = entry.get("description", "")
        if description and not (
            RULES["description_min"] <= len(description) <= RULES["description_max"]
        ):
            problems.append(
                f"'{page}' description has {len(description)} characters "
                f"(must be {RULES['description_min']} to {RULES['description_max']})"
            )
    return problems


def on_config(config):
    from mkdocs.exceptions import PluginError

    global PAGE_META
    try:
        PAGE_META = load()
    except ValueError as exc:
        raise PluginError(str(exc)) from exc
    return config


def on_files(files, config):
    pages = [
        file.src_uri
        for file in files.documentation_pages()
        if file.src_uri.endswith(PAGE_SUFFIXES)
    ]
    for problem in check(PAGE_META, pages):
        logger.warning(f"page_meta: {problem}")
    return files


def on_page_markdown(markdown, page, config, files):
    # Runs after `page.read_source`, which resets `page.meta` to the front
    # matter. mkdocs-jupyter renders notebooks later but calls this hook too.
    for key, value in PAGE_META.get(page.file.src_uri, {}).items():
        page.meta.setdefault(META_KEYS[key], value)
    _lowercase_canonical_url(page)
    return markdown


def _lowercase_canonical_url(page):
    """
    Write the path of the canonical URL of `page` in lowercase, the form that
    Netlify serves without a redirect (see the module docstring).
    """
    if page.canonical_url:
        parts = urlsplit(page.canonical_url)
        page.canonical_url = urlunsplit(parts._replace(path=parts.path.lower()))


def _pages_on_disk():
    """
    Documentation pages under docs/ (Markdown and notebooks), without the theme
    overrides, which `exclude_docs` keeps out of the site.
    """
    return [
        path.relative_to(DOCS_DIR).as_posix()
        for path in DOCS_DIR.rglob("*")
        if path.suffix in PAGE_SUFFIXES
        and "overrides" not in path.relative_to(DOCS_DIR).parts
        and ".ipynb_checkpoints" not in path.parts
    ]


if __name__ == "__main__":
    problems = check(load(), _pages_on_disk())
    for problem in problems:
        print(problem)
    print(f"{DATA_FILE.name}: {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)
