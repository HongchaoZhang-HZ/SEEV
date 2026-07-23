/*
 * Small, dependency-free copy control for SEEV documentation code blocks.
 *
 * For every Sphinx code block (``div.highlight`` containing a ``<pre>``), an
 * accessible button is injected that copies the block's text to the clipboard.
 * The button is keyboard reachable, exposes an aria-label, announces state via
 * aria-live, and falls back to a legacy copy path when the async Clipboard API
 * is unavailable. No third-party libraries are used.
 */
(function () {
  "use strict";

  function codeText(block) {
    var pre = block.querySelector("pre");
    if (!pre) {
      return "";
    }
    // Clone so the injected button text is never captured in the copy.
    var clone = pre.cloneNode(true);
    var btn = clone.querySelector(".seev-copy-button");
    if (btn) {
      btn.remove();
    }
    return clone.textContent.replace(/\n$/, "");
  }

  function writeClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      try {
        var area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.style.position = "absolute";
        area.style.left = "-9999px";
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        document.body.removeChild(area);
        resolve();
      } catch (err) {
        reject(err);
      }
    });
  }

  function decorate(block) {
    if (block.querySelector(".seev-copy-button")) {
      return;
    }
    var button = document.createElement("button");
    button.type = "button";
    button.className = "seev-copy-button";
    button.setAttribute("aria-label", "Copy code to clipboard");
    button.setAttribute("aria-live", "polite");
    button.textContent = "Copy";

    button.addEventListener("click", function () {
      writeClipboard(codeText(block)).then(
        function () {
          button.textContent = "Copied";
          button.setAttribute("data-copied", "true");
          button.setAttribute("aria-label", "Code copied to clipboard");
          window.setTimeout(function () {
            button.textContent = "Copy";
            button.removeAttribute("data-copied");
            button.setAttribute("aria-label", "Copy code to clipboard");
          }, 2000);
        },
        function () {
          button.textContent = "Error";
          window.setTimeout(function () {
            button.textContent = "Copy";
          }, 2000);
        }
      );
    });

    block.appendChild(button);
  }

  function init() {
    var blocks = document.querySelectorAll("div.highlight");
    for (var i = 0; i < blocks.length; i++) {
      decorate(blocks[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
