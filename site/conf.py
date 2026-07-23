# Configuration file for the SEEV project site.
#
# The site is a static, warning-clean Sphinx/Furo build. It documents the
# maintained focused CI path and the full research/certification path without
# inventing benchmark, verification, training, or reproducibility results. All
# project/paper metadata below is taken verbatim from the repository README.

project = "SEEV"
author = "Hongchao Zhang, Zhizhen Qin, Sicun Gao, Andrew Clark"
copyright = "2024, Hongchao Zhang, Zhizhen Qin, Sicun Gao, Andrew Clark"

# Full paper title, kept verbatim from README.md.
paper_title = (
    "SEEV: Synthesis with Efficient Exact Verification for ReLU Neural "
    "Barrier Functions"
)
# Venue and year, kept verbatim from README.md / the BibTeX entry.
venue = "The Thirty-eighth Annual Conference on Neural Information Processing Systems"
year = "2024"
paper_url = (
    "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
    "b7868dedad7192f83c9efb042da43317-Abstract-Conference.html"
)

# -- General configuration ---------------------------------------------------

extensions = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Do not fail the build on a missing/extra label; instead keep the build strict
# via the ``-W`` flag passed on the command line. Warnings are treated as
# errors in CI, so the source must remain clean.
nitpicky = False

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "SEEV"
html_static_path = ["_static"]
html_css_files = ["seev.css"]
html_js_files = ["copybutton.js"]
html_show_sourcelink = False
html_copy_source = False

# The coordinated light and dark palettes live in ``_static/seev.css``.
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#b54800",
        "color-brand-content": "#a84000",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ea6400",
        "color-brand-content": "#ea6400",
    },
    "sidebar_hide_name": False,
}
