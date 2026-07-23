(function () {
  "use strict";

  const pages = {
    "overview.html": [
      ["repository-map", "Repository map"],
      ["maintained-path-vs-research-path", "Maintained path vs. research path"],
    ],
    "method.html": [
      ["from-trained-network-to-certificate", "From trained network to certificate"],
      ["limitations-of-the-method", "Limitations of the method"],
    ],
    "getting-started.html": [
      ["focused-ci-path-python-3-10", "Focused CI path (Python 3.10+)"],
      ["full-research-certification-path", "Full research / certification path"],
    ],
    "usage.html": [
      ["focused-test-gate", "Focused test gate"],
      ["certification-example-darboux", "Certification example (Darboux)"],
      ["supported-system-names", "Supported system names"],
      ["training-command-files", "Training command files"],
    ],
    "limitations.html": [
      ["licensed-solvers", "Licensed solvers"],
      ["legacy-research-dependencies", "Legacy research dependencies"],
      ["pretrained-model-expectations", "Pretrained model expectations"],
      ["no-paper-scale-ci", "No paper-scale CI"],
      ["platform-and-resource-caveats", "Platform and resource caveats"],
    ],
    "citation.html": [["bibtex", "BibTeX"]],
  };

  function pageName(href) {
    const url = new URL(href, window.location.href);
    return url.pathname.split("/").pop() || "index.html";
  }

  function buildSectionTree(page, sections) {
    const list = document.createElement("ul");
    list.className = "seev-page-sections";
    list.hidden = true;
    list.setAttribute("aria-label", `Sections in ${page}`);

    for (const [id, label] of sections) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = `${page}#${id}`;
      link.textContent = label;
      link.dataset.sectionId = id;
      item.appendChild(link);
      list.appendChild(item);
    }
    return list;
  }

  function installNavigation() {
    const sidebar = document.querySelector(".sidebar-tree");
    if (!sidebar) return;

    let activeLinks = [];
    for (const pageLink of sidebar.querySelectorAll(".toctree-l1 > a")) {
      const page = pageName(pageLink.href);
      const sections = pages[page];
      if (!sections) continue;

      const item = pageLink.closest(".toctree-l1");
      const tree = buildSectionTree(page, sections);
      item.appendChild(tree);

      if (item.classList.contains("current-page")) {
        tree.hidden = false;
        pageLink.setAttribute("aria-expanded", "true");
        activeLinks = Array.from(tree.querySelectorAll("a"));
      } else {
        pageLink.setAttribute("aria-expanded", "false");
      }
    }

    if (!activeLinks.length) return;

    let framePending = false;
    const updateCurrentSection = function () {
      framePending = false;
      const candidates = activeLinks
        .map((link) => ({
          link,
          section: document.getElementById(link.dataset.sectionId),
        }))
        .filter(({ section }) => section);

      let current = null;
      for (const candidate of candidates) {
        if (candidate.section.getBoundingClientRect().top <= 160) {
          current = candidate;
        }
      }
      if (
        candidates.length &&
        window.scrollY + window.innerHeight >=
          document.documentElement.scrollHeight - 2
      ) {
        current = candidates[candidates.length - 1];
      }

      for (const candidate of candidates) {
        const isCurrent = candidate === current;
        candidate.link.classList.toggle("current-section", isCurrent);
        if (isCurrent) {
          candidate.link.setAttribute("aria-current", "location");
        } else {
          candidate.link.removeAttribute("aria-current");
        }
      }
    };

    const scheduleUpdate = function () {
      if (framePending) return;
      framePending = true;
      window.requestAnimationFrame(updateCurrentSection);
    };

    window.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("hashchange", scheduleUpdate);
    scheduleUpdate();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installNavigation);
  } else {
    installNavigation();
  }
})();
