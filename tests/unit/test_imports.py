"""Import isolation for the focused, license-free EEV surface."""

import os
import subprocess
import sys


def test_core_modules_import_without_solvers():
    from Modules.NNet import NeuralNetwork
    from Scripts.Status import NeuronStatus, NetworkStatus
    from Scripts.NSBasic import NSBasic, NS
    from EEV.Modules.Function import generate_samples

    assert callable(NeuralNetwork)
    assert callable(NeuronStatus)
    assert callable(NetworkStatus)
    assert callable(NSBasic)
    assert callable(NS)
    assert callable(generate_samples)


def test_core_import_succeeds_when_optional_solvers_are_blocked():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    eev_outer = os.path.join(root, "EEV")
    eev_inner = os.path.join(eev_outer, "EEV")
    code = """
import sys
sys.modules["pydrake"] = None
sys.modules["dreal"] = None
sys.modules["gurobipy"] = None
sys.modules["auto_LiRPA"] = None
from EEV.Modules.Function import generate_samples
from Scripts.NSBasic import NSBasic
from Scripts.Status import NeuronStatus
assert callable(generate_samples)
assert callable(NSBasic)
assert callable(NeuronStatus)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((eev_outer, eev_inner))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_optional_solver_import_does_not_hide_transitive_failure(tmp_path):
    fake_solver = tmp_path / "pydrake"
    fake_solver.mkdir()
    (fake_solver / "__init__.py").write_text("", encoding="utf-8")
    (fake_solver / "solvers.py").write_text(
        "import missing_solver_runtime_dependency\n", encoding="utf-8"
    )

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), os.path.join(root, "EEV"), os.path.join(root, "EEV", "EEV"))
    )
    result = subprocess.run(
        [sys.executable, "-c", "import EEV.Modules.Function"],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "missing_solver_runtime_dependency" in result.stderr


def test_parameter_import_does_not_load_case_studies():
    sys.modules.pop("Cases", None)
    sys.modules.pop("dreal", None)
    import Scripts.PARA as para

    assert para.zero_tol == 1e-16
    assert para.round_tol == 3
    assert "Cases" not in sys.modules
    assert "dreal" not in sys.modules
