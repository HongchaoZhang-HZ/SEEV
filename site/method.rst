Method
======

This page describes the verification pipeline as implemented in the ``EEV``
package, following ``certify_cbf.py`` and ``SearchVerifier`` /
``SearchVerifierMT``. It summarizes the source code; it does not restate or
add quantitative results from the paper.

From trained network to certificate
-----------------------------------

1. **Load a trained ReLU barrier.** ``certify_cbf.py`` reconstructs the barrier
   network as a ``NNet`` whose architecture is a linear input layer, the
   requested number of ReLU hidden layers of the requested width, and a single
   linear output. The trained ``state_dict`` from ``models/`` is loaded into
   this network.

2. **Bind the system case.** For the requested system name, the script builds
   the corresponding dynamics model and the matching ``EEV`` case
   (``Darboux``, ``ObsAvoid``, ``LinearSatellite``, or ``HighO``), which define
   the state space and the safe/unsafe sets used during verification.

3. **Seed the boundary search.** A safe sample and an unsafe sample are drawn
   and checked against the case's safe / unsafe masks. These seeds give the
   search a starting point on either side of the barrier's zero level set.

4. **Search activation regions along the boundary.** ``SearchVerifier`` walks
   the ReLU network's activation regions near the barrier boundary. Each
   candidate activation pattern defines a linear region; a linear-program
   feasibility check (``solver_lp``) tests whether that region is non-empty
   within the state space. Feasible regions are recorded, and their neighbours
   are enumerated (``Filter_S_neighbour`` / ``Possible_S``) and enqueued,
   giving a breadth-first exploration of the boundary.

5. **Verify each segment and hinge.** For each feasible boundary region the
   verifier runs a segment check (``seg_verification``); the search also tracks
   pair-wise and higher-order hinges (``pair_wise_hinge``, ``ho_hinge``) where
   activation regions meet. These checks establish the barrier condition
   exactly on the enumerated boundary rather than by sampling.

6. **Report pass or counterexample.** If every segment and hinge check passes,
   verification succeeds. Otherwise the verifier returns a concrete
   counterexample from the failing segment. ``certify_cbf.py`` prints the
   verification flag, the counterexample when present, the elapsed time, and an
   ``info`` dictionary that includes the number of boundary segments explored.

Limitations of the method
-------------------------

- The verification result applies to the exact trained network and case that
  are loaded; it does not generalize to other architectures or safety sets.
- Exact verification enumerates activation regions along the boundary, so its
  cost grows with network size and boundary complexity; the multi-process
  variant (``SearchVerifierMT``) exists to manage this cost.
- The certification path depends on the research dependencies and a licensed
  solver (see :doc:`limitations`); it is not part of the continuous-integration
  gate, and no results here are re-derived or benchmarked.
