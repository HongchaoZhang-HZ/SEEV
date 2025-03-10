from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import random

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import LinearSatellite
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash
from neural_clbf.controllers.controller_utils import merge_adjacent_linear_layers

import torch.nn as nn

import EEV
from EEV.Modules.NNet import NeuralNetwork as NNet
from EEV.SearchVerifierMT import SearchVerifierMT
from EEV.SearchVerifier import SearchVerifier


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 512
controller_period = 0.01

start_x = torch.tensor(
    [
        [0.5, 0.5, 0.0, -0.1, -0.1, 0.0],
    ]
)
simulation_dt = 0.01


def main(args):
    # Define the scenarios
    nominal_params = {
        "a": 500e3,
        "ux_target": 0.0,
        "uy_target": 0.0,
        "uz_target": 0.0,
    }
    scenarios = [
        nominal_params,
    ]

    # Define the dynamics model
    dynamics_model = LinearSatellite(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
        use_l1_norm=False,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-1.0, 1.0),  # x
        (-1.0, 1.0),  # y
        (-1.0, 1.0),  # z
        (-1.0, 1.0),  # xdot
        (-1.0, 1.0),  # ydot
        (-1.0, 1.0),  # zdot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=30000,
        val_split=0.1,
        batch_size=batch_size,
        quotas={"unsafe": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-1.5, 1.5), (-1.5, 1.5)],
        n_grid=25,
        x_axis_index=LinearSatellite.X,
        y_axis_index=LinearSatellite.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
    )
    rollout_state_space_experiment = RolloutStateSpaceExperiment(
        "Rollout State Space",
        start_x,
        plot_x_index=LinearSatellite.X,
        plot_x_label="$x$",
        plot_y_index=LinearSatellite.Y,
        plot_y_label="$y$",
        scenarios=[nominal_params],
        n_sims_per_start=1,
        t_sim=10.0,
    )
    experiment_suite = ExperimentSuite(
        [
            V_contour_experiment,
            rollout_state_space_experiment,
        ]
    )

    scale_parameter = 10.0
    cbf_hidden_layers = args.cbf_hidden_layers
    cbf_hidden_size = args.cbf_hidden_size

    regularize_boundary_pattern = args.sim_reg_weight != 0

    # Initialize the controller
    clbf_controller = NeuralCBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        cbf_hidden_layers=cbf_hidden_layers,
        cbf_hidden_size=cbf_hidden_size,
        cbf_lambda=0.1,
        controller_period=controller_period,
        cbf_relaxation_penalty=1e4,
        scale_parameter=scale_parameter,
        primal_learning_rate=1e-3,
        learn_shape_epochs=5,
        certification_starting_epoch=10,
        sloss_weight=1e2,
        uloss_weight=1e2,
        dsloss_weight=2.0,
        use_relu=True,
        # loss_threshold_until_certify=0.5,
        eps=0.2,
        boundary_regularize_interval=1,
        boundary_threshold=1.0,
        n_clusters=5,
        sigmoid_k_val=4,
        sim_reg_weight=args.sim_reg_weight,
        regularize_boundary_pattern=regularize_boundary_pattern,
        perform_certification=args.perform_certification,
        employ_ce=args.perform_certification,
    )

    model_name = (
        "linear_satellite_commit_{}_layers_{}_size_{}_seed_{}_certify_{}_reg_{}".format(
            current_git_hash(),
            cbf_hidden_layers,
            cbf_hidden_size,
            args.random_seed,
            args.perform_certification,
            args.sim_reg_weight,
        )
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/linear_satellite_cbf/relu",
        name=model_name,
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=101,
        deterministic=True,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)

    # Final Certification
    s_ub, s_lb = dynamics_model.state_limits
    normalization_A = torch.diag(scale_parameter * 2 / (s_ub - s_lb))
    normalization_b = ((s_ub * (-scale_parameter)) - (s_lb * scale_parameter)) / (
        s_ub - s_lb
    )
    normalization_layer = nn.Linear(
        in_features=normalization_A.shape[1],
        out_features=normalization_A.shape[0],
        bias=True,
    )
    normalization_layer.weight = nn.Parameter(torch.FloatTensor(normalization_A))
    normalization_layer.bias = nn.Parameter(torch.FloatTensor(normalization_b))

    v_nn = clbf_controller.V_nn
    v_nn_with_normalization = nn.Sequential(normalization_layer, *v_nn)
    v_nn_merged = merge_adjacent_linear_layers(v_nn_with_normalization)

    from EEV.Cases import LinearSat as LinearSat_EEV

    case = LinearSat_EEV()
    hdlayers = []
    for layer in range(cbf_hidden_layers):
        hdlayers.append(("relu", cbf_hidden_size))
    architecture = [("linear", 6)] + hdlayers + [("linear", 1)]
    model = NNet(architecture)
    trained_state_dict = v_nn_merged.state_dict()
    trained_state_dict = {
        f"layers.{key}": value for key, value in trained_state_dict.items()
    }
    model.load_state_dict(trained_state_dict, strict=True)

    safe_sample = dynamics_model.sample_safe(1)
    unsafe_sample = dynamics_model.sample_unsafe(1)
    while not all(dynamics_model.safe_mask(safe_sample)):
        safe_sample = dynamics_model.sample_safe(1)
    while not all(dynamics_model.unsafe_mask(unsafe_sample)):
        unsafe_sample = dynamics_model.sample_unsafe(1)
    spt = safe_sample.unsqueeze(0)
    uspt = unsafe_sample.unsqueeze(0)
    # Search Verification and output Counter Example
    # Search_prog = SearchVerifierMT(model, case)
    Search_prog = SearchVerifier(model, case)
    veri_flag, ce, info = Search_prog.SV_CE(spt, uspt)
    num_boundary_seg = info["num_boundary_seg"]
    if veri_flag:
        print("Verification successful!")
    else:
        print("Verification failed!")
        print("Counter example:", ce)

    CEs = clbf_controller.CEs
    CE_list = []
    for CE in CEs:
        ce_epoch, ce_point = CE
        if isinstance(ce_point, torch.Tensor):
            ce_point = ce_point.detach().cpu().numpy()
        if isinstance(ce_point, np.ndarray):
            ce_point = ce_point.tolist()

        CE_list.append((ce_epoch, ce_point))

    with open("result_{}.txt".format(model_name), "w") as f:
        import json

        results = {
            "veri_flag": veri_flag,
            "num_boundary_seg": num_boundary_seg,
            "early_stop_epoch": clbf_controller.early_stop_epoch,
            "CEs": CE_list,
        }
        json.dump(results, f)
    torch.save(v_nn_merged, "{}.pt".format(model_name))


if __name__ == "__main__":
    import argparse

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--cbf_hidden_layers", type=int, default=4)
    parser.add_argument("--cbf_hidden_size", type=int, default=8)
    parser.add_argument(
        "--perform_certification", type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--sim_reg_weight", type=float, default=0.0)
    args = parser.parse_args()

    # Set random seed
    seed = args.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    main(args)
