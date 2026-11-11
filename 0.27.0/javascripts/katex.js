// Render LaTeX math with KaTeX on every page, including after instant
// navigation. Notebooks (mkdocs-jupyter) keep their math between $...$ and
// $$...$$, and Markdown pages (pymdownx.arithmatex in generic mode) between
// \(...\) and \[...\]. Code blocks (pre, code) are never processed.
(function () {
  function render() {
    if (typeof renderMathInElement === "undefined") return;
    var content = document.querySelector(".md-content");
    if (!content) return;
    renderMathInElement(content, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(render);
  } else {
    document.addEventListener("DOMContentLoaded", render);
  }
})();
