// Draw the Mermaid diagrams of the Markdown pages (```mermaid fences, written
// as div.skf-mermaid by pymdownx.superfences, see mkdocs.yml) with the copy of
// Mermaid.js in docs/vendor/mermaid. Mermaid.js is loaded only on the pages
// that have a diagram, also after instant navigation, and the diagrams are
// drawn again when the reader switches between the light and dark schemes.
(function () {
  // Relative to this script, so it works in every version folder of the site
  var source = new URL(
    "../vendor/mermaid/mermaid.min.js",
    document.currentScript.src
  ).href;
  var loading = null;
  var counter = 0;

  function load() {
    if (!loading) {
      loading = new Promise(function (resolve, reject) {
        var script = document.createElement("script");
        script.src = source;
        script.onload = resolve;
        script.onerror = function () {
          loading = null;
          reject(new Error("could not load " + source));
        };
        document.head.appendChild(script);
      });
    }
    return loading;
  }

  function theme() {
    var scheme = document.body.getAttribute("data-md-color-scheme");
    return scheme === "slate" ? "dark" : "neutral";
  }

  function render() {
    var blocks = document.querySelectorAll("div.skf-mermaid");
    if (blocks.length === 0) return;
    load()
      .then(function () {
        window.mermaid.initialize({
          startOnLoad: false,
          theme: theme(),
          securityLevel: "strict"
        });
        // One diagram at a time; an invalid one does not stop the others
        var chain = Promise.resolve();
        blocks.forEach(function (block) {
          if (block.dataset.source === undefined) {
            block.dataset.source = block.textContent;
          }
          chain = chain.then(function () {
            return window.mermaid
              .render("skf-mermaid-" + counter++, block.dataset.source)
              .then(function (result) {
                block.innerHTML = result.svg;
                block.setAttribute("data-rendered", "");
              })
              .catch(function (error) {
                console.error("Mermaid:", error);
              });
          });
        });
        return chain;
      })
      .catch(function (error) {
        console.error("Mermaid:", error);
      });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(render);
  } else {
    document.addEventListener("DOMContentLoaded", render);
  }

  new MutationObserver(render).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"]
  });
})();
