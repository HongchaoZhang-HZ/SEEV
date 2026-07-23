"""End-to-end build check for the SEEV site.

The site is built once (warnings-as-errors) into a temporary directory and the
resulting HTML is inspected: every required page must be present and the local
CSS/JS assets must be emitted and referenced. Building here keeps the test
self-contained regardless of whether ``site/_build`` already exists.
"""

import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SITE = os.path.join(_ROOT, "site")

_REQUIRED_HTML = [
    "index.html",
    "overview.html",
    "method.html",
    "getting-started.html",
    "usage.html",
    "limitations.html",
    "citation.html",
]


@pytest.fixture(scope="module")
def built_html(tmp_path_factory):
    pytest.importorskip("furo")
    out = tmp_path_factory.mktemp("seev_html")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            _SITE,
            str(out),
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "sphinx build failed (warnings are errors):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return out


def test_all_required_pages_built(built_html):
    for name in _REQUIRED_HTML:
        assert os.path.isfile(os.path.join(built_html, name)), f"missing built page: {name}"


def test_local_assets_are_emitted(built_html):
    static = os.path.join(built_html, "_static")
    assert os.path.isfile(os.path.join(static, "seev.css"))
    assert os.path.isfile(os.path.join(static, "copybutton.js"))


def test_landing_references_local_assets(built_html):
    with open(os.path.join(built_html, "index.html"), "r", encoding="utf-8") as handle:
        html = handle.read()
    assert "seev.css" in html
    assert "copybutton.js" in html
    # The CSS-only hero must render as markup, not an external image.
    assert "seev-hero" in html


def test_landing_has_one_primary_heading(built_html):
    with open(os.path.join(built_html, "index.html"), "r", encoding="utf-8") as handle:
        html = handle.read()
    assert html.count("<h1") == 1


def test_citation_page_carries_bibtex(built_html):
    with open(os.path.join(built_html, "citation.html"), "r", encoding="utf-8") as handle:
        html = handle.read()
    assert "zhang2024seev" in html
