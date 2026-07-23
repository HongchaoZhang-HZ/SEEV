"""Contract tests for the SEEV Sphinx project site sources.

These checks read the ``site/`` sources as text (no Sphinx import required) and
assert that every required page exists, that the required commands and links are
present verbatim, that navigation wires every page into the table of contents,
and that the two-theme visual system, copy control, responsive rules, and
reduced-motion handling are all present.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SITE = os.path.join(_ROOT, "site")
_STATIC = os.path.join(_SITE, "_static")
_README = os.path.join(_ROOT, "README.md")

# The information architecture required by the task, keyed by source file.
_PAGES = {
    "index.rst": "landing",
    "overview.rst": "overview",
    "method.rst": "method",
    "getting-started.rst": "getting started",
    "usage.rst": "usage",
    "limitations.rst": "limitations",
    "citation.rst": "citation",
}


def _read(*parts):
    path = os.path.join(_SITE, *parts)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def test_all_required_pages_present():
    for name in _PAGES:
        assert os.path.isfile(os.path.join(_SITE, name)), f"missing page: {name}"


def test_conf_uses_furo_and_registers_local_assets():
    conf = _read("conf.py")
    assert 'html_theme = "furo"' in conf
    assert '"seev.css"' in conf
    assert '"copybutton.js"' in conf
    assert os.path.isfile(os.path.join(_STATIC, "seev.css"))
    assert os.path.isfile(os.path.join(_STATIC, "copybutton.js"))


def test_navigation_lists_every_page():
    index = _read("index.rst")
    assert ".. toctree::" in index
    for name in _PAGES:
        if name == "index.rst":
            continue
        doc = name[: -len(".rst")]
        assert doc in index, f"page not in toctree navigation: {doc}"


def test_landing_has_two_copyable_entry_commands():
    index = _read("index.rst")
    assert index.count(".. code-block::") >= 2
    assert "python -m pip install -r requirements-ci.txt" in index
    assert "python -m pytest tests/unit tests/ci -q" in index


def test_landing_has_three_capability_cards():
    index = _read("index.rst")
    assert index.count('class="seev-card"') == 3


def test_landing_has_paper_and_repository_links():
    index = _read("index.rst")
    assert (
        "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
        "b7868dedad7192f83c9efb042da43317-Abstract-Conference.html"
    ) in index
    assert "https://github.com/HongchaoZhang-HZ/SEEV" in index


def test_readme_links_public_documentation():
    with open(_README, "r", encoding="utf-8") as handle:
        readme = handle.read()
    assert "https://hongchaozhang-hz.github.io/SEEV/" in readme


def test_getting_started_distinguishes_paths_with_license_warning():
    text = _read("getting-started.rst")
    assert "Python 3.10" in text
    assert "requirements-ci.txt" in text
    assert ".. warning::" in text
    assert "Gurobi license" in text
    assert "requirements.txt" in text


def test_usage_has_exact_darboux_certification_example():
    text = _read("usage.rst")
    assert "cd neural_clbf_seev" in text
    assert (
        "python certify_cbf.py --system_name darboux --cbf_hidden_layers 2 "
        "--cbf_hidden_size 256 --model_path models/darboux_2_256.pt"
    ) in text


def test_usage_lists_supported_system_names_and_command_files():
    text = _read("usage.rst")
    for system in ("darboux", "obs_avoid", "linear_satellite", "high_o"):
        assert system in text
    assert "certify_commands.sh" in text
    assert "darboux_commands.txt" in text


def test_method_is_source_faithful_and_states_limits():
    text = _read("method.rst")
    for token in ("activation region", "boundary", "segment", "hinge", "counterexample"):
        assert token in text.lower(), f"method page missing: {token}"


def test_limitations_covers_required_caveats():
    text = _read("limitations.rst").lower()
    for token in ("licensed", "gurobi", "auto_lirpa", "pretrained", "continuous integration"):
        assert token in text, f"limitations page missing: {token}"


def test_citation_has_exact_bibtex():
    text = _read("citation.rst")
    assert "@inproceedings{zhang2024seev," in text
    assert (
        "author = {Zhang, Hongchao and Qin, Zhizhen and Gao, Sicun and "
        "Clark, Andrew}"
    ) in text
    assert "booktitle = {Advances in Neural Information Processing Systems}" in text
    assert "doi = {10.52202/079017-3214}" in text
    assert (
        "editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and "
        "U. Paquet and J. Tomczak and C. Zhang}"
    ) in text
    assert "pages = {101367--101392}" in text
    assert "publisher = {Curran Associates, Inc.}" in text
    assert (
        "title = {SEEV: Synthesis with Efficient Exact Verification for ReLU "
        "Neural Barrier Functions}"
    ) in text
    assert "volume = {37}" in text
    assert "year = {2024}" in text
    assert (
        "url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/"
        "b7868dedad7192f83c9efb042da43317-Paper-Conference.pdf}"
    ) in text


def test_readme_uses_official_proceedings_record_and_bibtex():
    with open(_README, "r", encoding="utf-8") as handle:
        readme = handle.read()
    assert (
        "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
        "b7868dedad7192f83c9efb042da43317-Abstract-Conference.html"
    ) in readme
    assert "@inproceedings{zhang2024seev," in readme
    assert "doi = {10.52202/079017-3214}" in readme
    assert "openreview.net" not in readme


def test_legacy_dependency_provenance_is_explicit():
    overview = _read("overview.rst")
    limitations = _read("limitations.rst")
    source = "https://github.com/HongchaoZhang-HZ/exactverif-reluncbf-nips23"
    assert source in overview
    assert source in limitations
    assert "not" in limitations.lower()
    assert "Gurobi" in limitations
