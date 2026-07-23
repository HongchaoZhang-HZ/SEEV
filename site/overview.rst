Overview
========

This page maps the repository and distinguishes the maintained focused path
from the full research path.

Repository map
--------------

``EEV/``
   The exact-verification Python package (installed as ``EEV``). It contains
   the neural-network model wrapper, the activation-region search, and the
   segment/hinge verifier used to certify a trained barrier function.

   - ``EEV/EEV/Modules/`` — the ReLU network wrapper (``NNet``) and supporting
     function utilities.
   - ``EEV/EEV/Verifier/`` — linear and non-linear verification routines and
     the SMT/LP solver interfaces.
   - ``EEV/EEV/Cases/`` — the case definitions (``Darboux``, ``ObsAvoid``,
     ``LinearSatellite``, ``HighO``) that describe each benchmark's state
     space and safety sets.
   - ``EEV/EEV/SearchVerifier.py`` and ``SearchVerifierMT.py`` — the
     boundary search coupled with verification (single-process and
     multi-threaded/multi-process variants).

``neural_clbf_seev/``
   Training and certification code adapted from
   `neural_clbf <https://github.com/MIT-REALM/neural_clbf>`_. It provides the
   dynamics models, the training command files, the pretrained ``models/``,
   and ``certify_cbf.py`` — the command-line entry point that loads a trained
   ReLU barrier and runs SEEV verification.

``tests/``
   The focused, license-free test surface:

   - ``tests/unit/`` and ``tests/ci/`` — the fast unit and CI-contract checks
     that form the maintained gate.
   - ``tests/performance/`` — a sample micro-benchmark.
   - ``tests/site/`` — contract tests for this documentation site.

``scripts/ci/``
   Strict CI helpers, including the JUnit report integrity checker and the
   Pillow remediation check.

Maintained path vs. research path
---------------------------------

**Maintained focused path.** The focused path targets Python 3.10+, installs
from ``requirements-ci.txt``, requires no licensed solver, and is exercised by
``tests/unit`` and ``tests/ci``. It is what the continuous-integration gate
runs. It covers the core data structures and the license-free portions of the
verification surface; it does not run paper-scale training or certification.

**Full research / certification path.** Reproducing the paper's training and
exact certification uses the dependencies pinned in ``requirements.txt``, an
`auto_LiRPA <https://github.com/Verified-Intelligence/auto_LiRPA>`_ setup,
editable installs of the ``EEV`` and ``neural_clbf_seev`` packages, and a
Gurobi license. This path is **not** covered by continuous integration. See
:doc:`getting-started` and :doc:`limitations` for the requirements and caveats.
