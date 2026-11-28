"""
MkDocs hook that corrects a bug of lunr.js in the search of the built site.

The search of Material for MkDocs runs lunr.js (2.3.9) in a worker. lunr keeps
the indexed terms in a graph and merges the nodes that are equal. To compare
two nodes it writes each one as text: every edge label followed by the id of
the node it leads to, with nothing in between. Labels are single characters
and ids are numbers, so two different nodes give the same text when a label is
a digit (label "1" to node 23 and label "1" to node 2 followed by label "3"
both start with "123"). The nodes are then merged by mistake and the graph
ends up with terms that are not in the index.

When a search expands to one of those terms (the last word of a query is
searched as a prefix, and so are the words that lunr filters as stop words,
such as "to", "should" or "how"), the worker raises "Cannot read properties of
undefined (reading '_index')" and the reader sees "No matching documents".
Whether it happens depends on the whole vocabulary of the site, so any change
of the documentation can trigger it or hide it: the texts of a pull request
with numbers such as "1e10" were enough to break every search with "to".

This hook adds separators to that text in the worker of the built site, which
removes the ambiguity. It warns if the code it replaces is not found (a new
version of Material or of lunr): check then whether the bug is still there,
and update or remove this hook.

Registered in `mkdocs.yml` as:

    hooks:
      - tools/docs/hooks/search_fix.py
"""

import logging
from pathlib import Path

logger = logging.getLogger("mkdocs.hooks.search_fix")

# Minified code of lunr.TokenSet.prototype.toString in the search worker of
# Material for MkDocs 9.7: `e` is the text of the node, `s` the label of an
# edge and `o` the node it leads to.
BUGGY = "e=e+s+o.id}return e}"
FIXED = 'e=e+s+"\\u0001"+o.id+"\\u0002"}return e}'


def on_post_build(config):
    workers = Path(config["site_dir"]) / "assets" / "javascripts" / "workers"
    files = sorted(workers.glob("search.*.min.js"))
    if not files:
        logger.warning("search_fix: no search worker found in %s", workers)
    for path in files:
        code = path.read_text(encoding="utf-8")
        if FIXED in code:
            continue
        if BUGGY not in code:
            logger.warning(
                "search_fix: the code of lunr.TokenSet.toString was not found in "
                "'%s', so the search of the site is not corrected. See "
                "tools/docs/hooks/search_fix.py",
                path.name,
            )
            continue
        path.write_text(code.replace(BUGGY, FIXED), encoding="utf-8")
