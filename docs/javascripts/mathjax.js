window.MathJax = {
  tex: {
    // 1. Agregamos soporte para '$' y '$$' que usan los notebooks
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // 2. Eliminamos o comentamos estas líneas para que MathJax
    // busque fórmulas en TODA la página, no solo en clases específicas.
    // ignoreHtmlClass: ".*|",
    // processHtmlClass: "arithmatex"
  }
};