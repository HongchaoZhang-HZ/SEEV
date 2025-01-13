<div align="center">

# SEEV: Synthesis with Efficient Exact Verification for ReLU Neural Barrier Functions

[![Conference](https://img.shields.io/badge/NeurIPS%20'24-Accepted-success)](https://openreview.net/forum?id=nWMqQHzI3W)

</div>

This repository contains the implementation of [SEEV (Synthesis with Efficient Exact Verification)](https://openreview.net/forum?id=nWMqQHzI3W), a novel framework for synthesizing Neural Control Barrier Functions (NCBFs) with ReLU activations and performing efficient safety verification. The SEEV approach integrates synthesis and verification to reduce computational overhead while maintaining safety guarantees for autonomous systems. It includes algorithms for training NCBFs with regularization and efficient verification of safety conditions across benchmark systems.

## Requirements

To install requirements and set up for the project:

```setup
pip install -r requirements.txt
[Inside NCBCV] pip install -e .
[Inside neural_clbf_ncbcv] pip install -e .
```

Note that the directory `neural_clbf_ncbcv` is adapted from https://github.com/MIT-REALM/neural_clbf. 

## Training the Neural Control Barrier Functions (NCBF)

The commands for trainig the CBFs are located in `neural_clbf_ncbcv/darboux_commands.txt`, `neural_clbf_ncbcv/obs_avoid_commands.txt`, `neural_clbf_ncbcv/linear_satellite_commands.txt` and `neural_clbf_ncbcv/high_o_commands.txt`. The commands in these files are properly seeded, with hyperparameters specified accordingly.

## Evaluating the NCBF Models

To perform certification, run the commands located in `neural_clbf_ncbcv/certify_commands.sh`, which evaluates pretrained models located in `neural_clbf_ncbcv/models`. The metrics reported in the paper will be outputs to stdout.

## Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@inproceedings{
zhang2024seev,
title={{SEEV}: Synthesis with Efficient Exact Verification for Re{LU} Neural Barrier Functions},
author={Zhang, Hongchao and Qin, Zhizhen and Gao, Sicun and Clark, Andrew},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=nWMqQHzI3W}
}
