Usage
=====

This page covers the focused test gate, the exact certification example, the
supported system names, and where the training command files live.

Focused test gate
-----------------

The maintained gate installs the license-free dependencies and runs the
focused unit and CI-contract tests:

.. code-block:: bash

   python -m pip install -r requirements-ci.txt
   python -m pytest tests/unit tests/ci -q

Certification example (Darboux)
-------------------------------

Certification runs from inside the ``neural_clbf_seev`` directory and evaluates
a pretrained model from ``neural_clbf_seev/models``. The exact commands live in
``neural_clbf_seev/certify_commands.sh``. For the Darboux system:

.. code-block:: bash

   cd neural_clbf_seev
   python certify_cbf.py --system_name darboux --cbf_hidden_layers 2 --cbf_hidden_size 256 --model_path models/darboux_2_256.pt

The metrics reported in the paper are written to stdout. This example requires
the full research / certification path from :doc:`getting-started`, including a
Gurobi license; it is not part of continuous integration.

Supported system names
-----------------------

``certify_cbf.py`` accepts the following ``--system_name`` values:

- ``darboux``
- ``obs_avoid``
- ``linear_satellite``
- ``high_o``

Each also requires ``--cbf_hidden_layers``, ``--cbf_hidden_size``, and
``--model_path`` matching a pretrained model in ``neural_clbf_seev/models``.
The full set of certification commands for every system is in
``neural_clbf_seev/certify_commands.sh``.

Training command files
----------------------

The seeded training commands, with hyperparameters specified per run, live in
``neural_clbf_seev``:

- ``neural_clbf_seev/darboux_commands.txt``
- ``neural_clbf_seev/obs_avoid_commands.txt``
- ``neural_clbf_seev/linear_satellite_commands.txt``
- ``neural_clbf_seev/high_o_commands.txt``

Training and certification use the full research path and its licensed-solver
requirement; see :doc:`limitations`.
