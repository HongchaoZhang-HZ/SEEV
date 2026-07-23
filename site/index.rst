SEEV
====

.. raw:: html

   <div class="seev-hero" role="presentation">
     <div class="seev-hero__grid"></div>
     <div class="seev-hero__glow"></div>
     <span class="seev-tagline">NeurIPS 2024</span>
     <p class="seev-hero__title">Synthesis with Efficient Exact Verification<br>for ReLU Neural Barrier Functions</p>
     <p>SEEV integrates the synthesis and exact verification of ReLU Neural
     Control Barrier Functions so that safety conditions can be checked over
     activation regions without discarding formal guarantees.</p>
   </div>

SEEV is the reference implementation of `SEEV: Synthesis with Efficient Exact
Verification for ReLU Neural Barrier Functions
<https://openreview.net/forum?id=nWMqQHzI3W>`_, published at the
Thirty-eighth Annual Conference on Neural Information Processing Systems
(NeurIPS 2024). The framework trains ReLU Neural Control Barrier Functions
(NCBFs) with verification-aware regularization and then verifies the safety
conditions exactly by reasoning over the network's activation regions.

- Paper and reviews: https://openreview.net/forum?id=nWMqQHzI3W
- Source repository: https://github.com/HongchaoZhang-HZ/SEEV — the ``EEV``
  verification package and the ``neural_clbf_seev`` training/certification
  code.

Get started
-----------

The focused, license-free path targets **Python 3.10+** and installs from
``requirements-ci.txt``. Install the focused dependencies:

.. code-block:: bash

   python -m pip install -r requirements-ci.txt

Then run the focused test gate:

.. code-block:: bash

   python -m pytest tests/unit tests/ci -q

The full research and certification path (paper-scale training and exact
certification) has additional dependencies and a licensed-solver requirement;
see :doc:`getting-started` and :doc:`limitations`.

What SEEV provides
------------------

.. raw:: html

   <div class="seev-cards">
     <div class="seev-card">
       <h3>ReLU NCBF synthesis</h3>
       <p>Trains ReLU Neural Control Barrier Functions with
       verification-aware regularization for benchmark control systems.</p>
     </div>
     <div class="seev-card">
       <h3>Exact boundary verification</h3>
       <p>Enumerates activation regions along the barrier boundary and checks
       each segment and hinge exactly, rather than by sampling.</p>
     </div>
     <div class="seev-card">
       <h3>Counterexample output</h3>
       <p>Reports a pass or returns a concrete counterexample from segment
       verification, so failures are actionable.</p>
     </div>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   overview
   method
   getting-started
   usage
   limitations
   citation
