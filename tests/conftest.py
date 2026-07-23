"""Shared path setup for the focused, license-free EEV test surface.

The EEV package historically relies on ``sys.path`` manipulation and a mix of
``EEV.``-prefixed and bare (``Modules``/``Scripts``/``Cases``) imports. To make
the core data and neural-status behaviour importable from the ``tests`` tree
without installing the package, both the outer project directory (so ``import
EEV`` resolves) and the inner package directory (so the bare imports resolve)
are placed on ``sys.path``. The strict CI helpers under ``scripts/ci`` are made
importable the same way.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Outer directory: exposes the ``EEV`` package for ``import EEV`` / ``from EEV...``.
_EEV_OUTER = os.path.join(_ROOT, "EEV")
# Inner directory: exposes bare ``Modules`` / ``Scripts`` / ``Cases`` imports
# used internally by the EEV modules.
_EEV_INNER = os.path.join(_EEV_OUTER, "EEV")
# Strict CI helpers exercised by ``tests/ci``.
_SCRIPTS_CI = os.path.join(_ROOT, "scripts", "ci")

for _path in (_EEV_OUTER, _EEV_INNER, _SCRIPTS_CI):
    if _path not in sys.path:
        sys.path.insert(0, _path)
