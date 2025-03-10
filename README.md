<div align="center">

# SEEV: Synthesis with Efficient Exact Verification for ReLU Neural Barrier Functions

[![Conference](https://img.shields.io/badge/NeurIPS%20'24-Accepted-success)](https://openreview.net/forum?id=nWMqQHzI3W)

</div>

This repository contains the implementation of [SEEV (Synthesis with Efficient Exact Verification)](https://openreview.net/forum?id=nWMqQHzI3W), a novel framework for synthesizing Neural Control Barrier Functions (NCBFs) with ReLU activations and performing efficient safety verification. The SEEV approach integrates synthesis and verification to reduce computational overhead while maintaining safety guarantees for autonomous systems. It includes algorithms for training NCBFs with regularization and efficient verification of safety conditions across benchmark systems.

## Requirements

The requirements has been tested for python version 3.9.

To install requirements and set up for the project, first, install [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) following the project's README.

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
@inproceedings{
zhang2024seev,
title={{SEEV}: Synthesis with Efficient Exact Verification for Re{LU} Neural Barrier Functions},
author={Hongchao Zhang and Zhizhen Qin and Sicun Gao and Andrew Clark},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=nWMqQHzI3W}
}
