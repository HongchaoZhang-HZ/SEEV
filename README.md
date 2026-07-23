<div align="center">

# SEEV: Synthesis with Efficient Exact Verification for ReLU Neural Barrier Functions

[![Conference](https://img.shields.io/badge/NeurIPS%20'24-Accepted-success)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b7868dedad7192f83c9efb042da43317-Abstract-Conference.html)

</div>

This repository contains the implementation of [SEEV (Synthesis with Efficient Exact Verification)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b7868dedad7192f83c9efb042da43317-Abstract-Conference.html), a novel framework for synthesizing Neural Control Barrier Functions (NCBFs) with ReLU activations and performing efficient safety verification. The SEEV approach integrates synthesis and verification to reduce computational overhead while maintaining safety guarantees for autonomous systems. It includes algorithms for training NCBFs with regularization and efficient verification of safety conditions across benchmark systems.

📖 **Documentation:** https://hongchaozhang-hz.github.io/SEEV/ — build locally with `python -m sphinx -W --keep-going -b html site site/_build/html`.

## Quick start (focused path)

The maintained, license-free path targets **Python 3.10+** and installs from `requirements-ci.txt`:

```bash
python -m pip install -r requirements-ci.txt
python -m pytest tests/unit tests/ci -q
```

The full research and certification path below requires additional dependencies and a Gurobi license.

## Requirements

The full research and certification path has been tested for Python 3.9; the focused CI path above targets Python 3.10+.

The `auto_LiRPA` search integration is inherited from the legacy
[`exactverif-reluncbf-nips23`](https://github.com/HongchaoZhang-HZ/exactverif-reluncbf-nips23)
verification path. Gurobi enters through the adapted
[`neural_clbf`](https://github.com/MIT-REALM/neural_clbf) training stack; it is
not declared by `exactverif-reluncbf-nips23`. Both are legacy, optional
integrations outside the focused CI path.

To install requirements and set up for the full legacy path, first install
[auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) following the
project's README.

Then, run the following commands:

```setup
pip install -r requirements.txt
[Inside EEV] pip install -e .
[Inside neural_clbf_seev] pip install -e .
```

A gurobi license is required.


Note that the directory `neural_clbf_seev` is adapted from https://github.com/MIT-REALM/neural_clbf. 

## Training the Neural Control Barrier Functions (NCBF)

The commands for trainig the CBFs are located in `neural_clbf_seev/darboux_commands.txt`, `neural_clbf_seev/obs_avoid_commands.txt`, `neural_clbf_seev/linear_satellite_commands.txt` and `neural_clbf_seev/high_o_commands.txt`. The commands in these files are properly seeded, with hyperparameters specified accordingly.

## Evaluating the NCBF Models

To perform certification, run the commands located in `neural_clbf_seev/certify_commands.sh`, which evaluates pretrained models located in `neural_clbf_seev/models`. The metrics reported in the paper will be outputs to stdout.

## Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@inproceedings{zhang2024seev,
    author = {Zhang, Hongchao and Qin, Zhizhen and Gao, Sicun and Clark, Andrew},
    booktitle = {Advances in Neural Information Processing Systems},
    doi = {10.52202/079017-3214},
    editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
    pages = {101367--101392},
    publisher = {Curran Associates, Inc.},
    title = {SEEV: Synthesis with Efficient Exact Verification for ReLU Neural Barrier Functions},
    url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/b7868dedad7192f83c9efb042da43317-Paper-Conference.pdf},
    volume = {37},
    year = {2024}
}
```
