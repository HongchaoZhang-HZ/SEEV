Limitations
===========

SEEV is a research artifact. The following limitations are intrinsic to how the
repository is built and maintained, and they should be read before relying on
any part of it.

Licensed solvers
----------------

The adapted ``neural_clbf`` training stack includes a Gurobi evaluation
backend. It is a legacy optional integration and requires a **Gurobi license**.
The upstream `exactverif-reluncbf-nips23
<https://github.com/HongchaoZhang-HZ/exactverif-reluncbf-nips23>`_ repository
does not declare or import Gurobi.

Legacy research dependencies
----------------------------

The research path pins older libraries — for example ``torch==1.9.1``,
``pytorch-lightning==1.3.4``, ``cvxpy==1.2.1``, and ``dreal==4.21.6.2`` — and
also contains the ``auto_LiRPA`` search integration inherited from
`exactverif-reluncbf-nips23
<https://github.com/HongchaoZhang-HZ/exactverif-reluncbf-nips23>`_. These
legacy dependencies can conflict with newer toolchains and are separate from
the license-free ``requirements-ci.txt`` used by the focused path.

Pretrained model expectations
-----------------------------

``certify_cbf.py`` loads a pretrained checkpoint whose architecture must match
the ``--cbf_hidden_layers`` and ``--cbf_hidden_size`` arguments. The provided
checkpoints live in ``neural_clbf_seev/models``; certifying a different network
requires a compatible checkpoint that you supply.

No paper-scale CI
-----------------

Continuous integration runs only the focused, license-free surface
(``tests/unit`` and ``tests/ci``). It does **not** run training, certification,
or any paper-scale reproduction, and this documentation does not re-derive,
benchmark, or restate the paper's quantitative results.

Platform and resource caveats
-----------------------------

Exact verification enumerates activation regions along the barrier boundary, so
runtime and memory grow with network size and boundary complexity. The
multi-process variant (``SearchVerifierMT``) uses multiprocessing and therefore
benefits from multiple cores. The research dependencies and solver may further
constrain the operating systems and hardware on which the full path runs.
