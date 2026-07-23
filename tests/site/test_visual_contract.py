"""Contract tests for the SEEV documentation visual and interaction system.

The dark palette, single orange accent, code-block left rule, responsive
layout, and reduced-motion handling live in ``_static/seev.css``. The
dependency-free copy control lives in ``_static/copybutton.js``. These checks
assert their presence as text so no browser or bundler is required.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_STATIC = os.path.join(_ROOT, "site", "_static")


def _read(name):
    with open(os.path.join(_STATIC, name), "r", encoding="utf-8") as handle:
        return handle.read()


def test_dark_palette_and_single_accent():
    css = _read("seev.css").lower()
    assert "#09090b" in css  # dark canvas
    assert "#16161a" in css  # dark cards
    assert "#ea6400" in css  # single orange accent


def test_readable_content_width_around_820px():
    css = _read("seev.css")
    assert "max-width: 820px" in css


def test_code_block_has_orange_left_rule():
    css = _read("seev.css")
    assert "border-left: 3px solid var(--color-brand-primary)" in css


def test_hero_texture_is_css_only():
    css = _read("seev.css")
    assert ".seev-hero__grid" in css
    assert "linear-gradient" in css
    assert "radial-gradient" in css
    # No raster or generated SVG assets should ship with the site.
    for name in os.listdir(_STATIC):
        assert not name.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".gif"))


def test_responsive_and_reduced_motion_rules_present():
    css = _read("seev.css")
    assert "@media (max-width:" in css
    assert "prefers-reduced-motion: reduce" in css


def test_strong_keyboard_focus_style():
    css = _read("seev.css")
    assert ":focus-visible" in css
    assert "outline" in css


def test_copy_control_is_accessible_and_dependency_free():
    js = _read("copybutton.js")
    assert "seev-copy-button" in js
    assert "aria-label" in js
    assert "clipboard" in js.lower()
    # Keep the control dependency-free: no imports or external URLs.
    assert "import " not in js
    assert "http://" not in js and "https://" not in js
