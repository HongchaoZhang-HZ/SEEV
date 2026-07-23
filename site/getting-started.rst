Getting started
===============

There are two ways to use this repository: the maintained **focused CI path**
and the full **research / certification path**. Choose based on whether you
need paper-scale training and exact certification.

Focused CI path (Python 3.10+)
------------------------------

The focused path is license-free and is the one exercised by continuous
integration. It requires **Python 3.10 or newer**.

Install the focused dependencies:

.. code-block:: bash

   python -m pip install -r requirements-ci.txt

Run the focused test gate:

.. code-block:: bash

   python -m pytest tests/unit tests/ci -q

This installs only the license-free dependencies pinned in
``requirements-ci.txt`` and does not require a solver license. It covers the
core data structures and license-free verification surface; it does **not** run
paper-scale training or certification.

Full research / certification path
----------------------------------

.. warning::

   The full path is **not** covered by continuous integration. It requires
   additional, older pinned dependencies (see ``requirements.txt``) and a
   **Gurobi license**. Reproducing it may require a specific platform and
   non-trivial resources; results are not re-derived on this site.

Reproducing the paper's training and exact certification requires:

1. Installing the legacy
   `auto_LiRPA <https://github.com/Verified-Intelligence/auto_LiRPA>`_
   integration inherited from
   `exactverif-reluncbf-nips23
   <https://github.com/HongchaoZhang-HZ/exactverif-reluncbf-nips23>`_.
2. Installing the pinned research dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Editable installs of the two in-repository packages:

   .. code-block:: bash

      cd EEV && pip install -e .
      cd neural_clbf_seev && pip install -e .

4. A valid **Gurobi license** for the optional evaluation backend inherited
   through the adapted
   `neural_clbf <https://github.com/MIT-REALM/neural_clbf>`_ stack.

The ``neural_clbf_seev`` directory is adapted from
`neural_clbf <https://github.com/MIT-REALM/neural_clbf>`_. Once this path is
installed, continue to :doc:`usage` for the exact certification example and the
supported system names.
