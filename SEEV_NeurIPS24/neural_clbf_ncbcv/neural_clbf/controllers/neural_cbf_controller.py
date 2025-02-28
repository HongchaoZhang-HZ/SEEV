import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict
import random

from sklearn.cluster import KMeans
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.cbf_controller import CBFController
from neural_clbf.controllers.controller_utils import normalize_with_angles
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.controllers.controller_utils import merge_adjacent_linear_layers


import NCBCV
from NCBCV.Modules.NNet import NeuralNetwork as NNet
from NCBCV.SearchVerifierMT import SearchVerifierMT
from NCBCV.SearchVerifier import SearchVerifier


def sigmoid_k(x, k=1):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x).float()
    return 1 / (1 + torch.exp(-k * x))


class NeuralCBFController(pl.LightningModule, CBFController):
    """
    A neural CBF controller. Differs from the CBFController in that it uses a
    neural network to learn the CBF.

    More specifically, the CBF controller looks for a V such that

    V(safe) < 0
    V(unsafe) > 0
    dV/dt <= -lambda V

    This proves forward invariance of the 0-sublevel set of V, and since the safe set is
    a subset of this sublevel set, we prove that the unsafe region is not reachable from
    the safe region.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        datamodule: EpisodicDataModule,
        experiment_suite: ExperimentSuite,
        cbf_hidden_layers: int = 2,
        cbf_hidden_size: int = 48,
        cbf_lambda: float = 1.0,
        cbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        scale_parameter: float = 10.0,
        learn_shape_epochs: int = 0,
        use_relu: bool = False,
        num_representative_samples=10000,
        boundary_regularize_interval=1,
        boundary_threshold=1.0,
        n_clusters=10,
        sigmoid_k_val=4,
        sim_reg_weight=0.05,
        regularize_boundary_pattern=False,
        sloss_weight=1e2,
        uloss_weight=1e2,
        loss_threshold_until_certify=3e-2,
        dsloss_weight=1.0,
        eps=1e-2,
        certification_starting_epoch=0,
        employ_ce=False,
        perform_certification=False,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            cbf_hidden_layers: number of hidden layers to use for the CLBF network
            cbf_hidden_size: number of neurons per hidden layer in the CLBF network
            cbf_lambda: convergence rate for the CLBF
            cbf_relaxation_penalty: the penalty for relaxing CLBF conditions.
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the CLBF decrease loss
            scale_parameter: normalize non-angle data points to between +/- this value.
            learn_shape_epochs: number of epochs to spend just learning the shape
            use_relu: if True, use a ReLU network instead of Tanh
        """
        super(NeuralCBFController, self).__init__(
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            experiment_suite=experiment_suite,
            cbf_lambda=cbf_lambda,
            cbf_relaxation_penalty=cbf_relaxation_penalty,
            controller_period=controller_period,
        )
        self.save_hyperparameters()

        # Save the provided model
        # self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the datamodule
        self.datamodule = datamodule

        # Save the experiments suits
        self.experiment_suite = experiment_suite

        # Save the other parameters
        self.primal_learning_rate = primal_learning_rate
        self.learn_shape_epochs = learn_shape_epochs

        # Compute and save the center and range of the state variables
        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        self.k = scale_parameter
        self.x_range = self.x_range / self.k
        # We shouldn't scale or offset any angle dimensions
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_range[self.dynamics_model.angle_dims] = 1.0

        # Some of the dimensions might represent angles. We want to replace these
        # dimensions with two dimensions: sin and cos of the angle. To do this, we need
        # to figure out how many numbers are in the expanded state
        n_angles = len(self.dynamics_model.angle_dims)
        self.n_dims_extended = self.dynamics_model.n_dims + n_angles

        # Define the CLBF network, which we denote V
        self.cbf_hidden_layers = cbf_hidden_layers
        self.cbf_hidden_size = cbf_hidden_size
        activation = nn.ReLU() if use_relu else nn.Tanh()
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended, self.cbf_hidden_size
        )
        self.V_layers["input_activation"] = activation
        for i in range(self.cbf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.cbf_hidden_size, self.cbf_hidden_size
            )
            if i < self.cbf_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = activation
        self.V_layers["output_linear"] = nn.Linear(self.cbf_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)
        self.num_representative_samples = num_representative_samples
        self.rs = self.dynamics_model.sample_state_space(
            self.num_representative_samples
        )
        self.boundary_regularize_interval = boundary_regularize_interval
        self.boundary_threshold = boundary_threshold
        self.n_clusters = n_clusters
        self.sigmoid_k_val = sigmoid_k_val
        self.sim_reg_weight = sim_reg_weight
        self.regularize_boundary_pattern = regularize_boundary_pattern

        self.sloss_weight = sloss_weight
        self.uloss_weight = uloss_weight
        self.dsloss_weight = dsloss_weight

        self.eps = eps

        self.loss_threshold_until_certify = loss_threshold_until_certify

        self.early_stop = False
        self.early_stop_epoch = -1
        self.CEs = []

        self.certification_starting_epoch = certification_starting_epoch
        self.num_boundary_seg = None

        self.employ_ce = employ_ce
        self.perform_certification = perform_certification

    @property
    def cbf_lambda(self):
        """Rename clf lambda to cbf"""
        return self.clf_lambda

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        return self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # Apply the offset and range to normalize about zero
        x_norm = normalize_with_angles(self.dynamics_model, x, k=self.k)

        # Compute the CLBF layer-by-layer, computing the Jacobian alongside

        # We need to initialize the Jacobian to reflect the normalization that's already
        # been done to x
        bs = x_norm.shape[0]
        JV = torch.zeros(
            (bs, self.n_dims_extended, self.dynamics_model.n_dims)
        ).type_as(x)
        # and for each non-angle dimension, we need to scale by the normalization
        for dim in range(self.dynamics_model.n_dims):
            JV[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)

        # And adjust the Jacobian for the angle dimensions
        for offset, sin_idx in enumerate(self.dynamics_model.angle_dims):
            cos_idx = self.dynamics_model.n_dims + offset
            JV[:, sin_idx, sin_idx] = x_norm[:, cos_idx]
            JV[:, cos_idx, sin_idx] = -x_norm[:, sin_idx]

        # Now step through each layer in V
        V = x_norm
        for layer in self.V_nn:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V**2), JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)

        return V, JV

    def forward(self, x):
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        return self.u(x)

    def boundary_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
        is_ce: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = self.eps
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        V = self.V(x)

        #   2.) V < 0 in the safe region
        V_safe = V[safe_mask]
        safe_violation = F.relu(eps + V_safe)
        safe_V_term = self.sloss_weight * safe_violation.mean()
        if is_ce:
            loss.append(("CE CLBF safe region term", safe_V_term))
        else:
            loss.append(("CLBF safe region term", safe_V_term))
        if accuracy:
            safe_V_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
            loss.append(("CLBF safe region accuracy", safe_V_acc))

        #   3.) V > 0 in the unsafe region
        V_unsafe = V[unsafe_mask]
        unsafe_violation = F.relu(eps - V_unsafe)
        unsafe_V_term = self.uloss_weight * unsafe_violation.mean()
        if is_ce:
            loss.append(("CE CLBF unsafe region term", unsafe_V_term))
        else:
            loss.append(("CLBF unsafe region term", unsafe_V_term))

        if accuracy:
            unsafe_V_acc = (unsafe_violation <= eps).sum() / unsafe_violation.nelement()
            loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))

        return loss

    def descent_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
        requires_grad: bool = False,
        is_ce: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to the descent condition

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        # We'll encourage satisfying the CBF conditions by...
        #
        #   1) Minimize the relaxation needed to make the QP feasible.

        # Get the control input and relaxation from solving the QP, and aggregate
        # the relaxation across scenarios
        u_qp, qp_relaxation = self.solve_CLF_QP(x, requires_grad=requires_grad)
        qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        # Minimize the qp relaxation to encourage satisfying the decrease condition
        qp_relaxation_loss = self.dsloss_weight * qp_relaxation.mean()
        if is_ce:
            loss.append(("CE QP relaxation", qp_relaxation_loss))
        else:
            loss.append(("QP relaxation", qp_relaxation_loss))

        return loss

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.early_stop:
            return -1

    def boundary_regularization_loss(self) -> List[Tuple[str, torch.Tensor]]:
        loss = []
        x = self.rs

        with torch.no_grad():
            sample_outs = self.V(x)

        near_zero_mask = (torch.abs(sample_outs) < self.boundary_threshold).squeeze()

        near_zero_samples_np = x[near_zero_mask].detach().cpu().numpy()
        if near_zero_samples_np.shape[0] < self.n_clusters:
            return loss
        kmeans = KMeans(
            n_clusters=self.n_clusters, n_init="auto", init="k-means++"
        ).fit(near_zero_samples_np)
        samples_by_cluster = []
        for cluster_idx in range(self.n_clusters):
            clustered_samples = near_zero_samples_np[kmeans.labels_ == cluster_idx]
            samples_by_cluster.append(clustered_samples)

        avg_sim_regs = []
        for cluster in samples_by_cluster:
            num_samples_cluster = cluster.shape[0]
            if num_samples_cluster == 0:
                continue

            cluster_in = torch.from_numpy(cluster).float()
            cluster_in_normalize = normalize_with_angles(
                self.dynamics_model, cluster_in, k=self.k
            )
            input_to_relu1 = self.V_nn[0](cluster_in_normalize)
            approx_pattern1 = sigmoid_k(input_to_relu1, self.sigmoid_k_val)
            relu1_output = self.V_nn[1](input_to_relu1)

            input_to_relu2 = self.V_nn[2](relu1_output)
            approx_pattern2 = sigmoid_k(input_to_relu2, self.sigmoid_k_val)
            # relu2_output = self.V_nn[3](input_to_relu2)

            approx_pattern = torch.cat((approx_pattern1, approx_pattern2), dim=1)
            idx = torch.randperm(approx_pattern.shape[0])
            selected_near_zero_approx_pattern = approx_pattern[idx]

            diff = selected_near_zero_approx_pattern.unsqueeze(
                1
            ) - selected_near_zero_approx_pattern.unsqueeze(0)

            dist = torch.sum(torch.abs(diff), dim=2)
            sum_distances = dist.sum()
            avg_sim_regs.append(
                sum_distances / (num_samples_cluster * num_samples_cluster)
            )

        if len(avg_sim_regs) == 0:
            sim_reg = 0
        else:
            sim_reg = sum(avg_sim_regs) / len(avg_sim_regs)

        sim_reg = sim_reg * self.sim_reg_weight
        loss.append(("Boundary Regularization", sim_reg))
        print("Boundary Regularization: ", sim_reg)
        return loss

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Compute the losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask)
        )
        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(
                    x, goal_mask, safe_mask, unsafe_mask, requires_grad=True
                )
            )

        if self.employ_ce and len(self.CEs) > 0:
            CEs = [ce[1] for ce in self.CEs]
            CEs = torch.FloatTensor(CEs).to(self.device)
            ce_goal_mask = self.dynamics_model.goal_mask(CEs)
            ce_safe_mask = self.dynamics_model.safe_mask(CEs)
            ce_unsafe_mask = self.dynamics_model.unsafe_mask(CEs)
            component_losses.update(
                self.boundary_loss(
                    CEs, ce_goal_mask, ce_safe_mask, ce_unsafe_mask, is_ce=True
                )
            )
            if self.current_epoch > self.learn_shape_epochs:
                component_losses.update(
                    self.descent_loss(
                        CEs,
                        ce_goal_mask,
                        ce_safe_mask,
                        ce_unsafe_mask,
                        is_ce=True,
                        requires_grad=True,
                    )
                )

        if (
            self.regularize_boundary_pattern
            and self.current_epoch != 0
            and self.current_epoch % self.boundary_regularize_interval == 0
            and batch_idx == 0
        ):
            # if self.current_epoch % self.boundary_regularize_interval == 0:
            component_losses.update(self.boundary_regularization_loss())

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        batch_dict = {"loss": total_loss, **component_losses}

        return batch_dict

    def ready_to_certify(self):
        if not self.perform_certification:
            return False
        if self.current_epoch < self.certification_starting_epoch:
            return False
        
        safe_mask = self.dynamics_model.safe_mask(self.rs)
        unsafe_mask = self.dynamics_model.unsafe_mask(self.rs)

        safe_samples = self.rs[safe_mask]
        unsafe_samples = self.rs[unsafe_mask]

        safe_outs = self.V(safe_samples)
        unsafe_outs = self.V(unsafe_samples)

        num_safe_violations = torch.sum(safe_outs > 0)
        num_unsafe_violations = torch.sum(unsafe_outs < 0)
        print("Num safe violations: ", num_safe_violations)
        print("Num unsafe violations: ", num_unsafe_violations)

        # NOTE: Hardcoding
        return num_unsafe_violations == 0 and num_safe_violations < 16

    def training_epoch_end(self, outputs):
        if self.early_stop:
            return
        """This function is called after every epoch is completed."""
        # Outputs contains a list for each optimizer, and we need to collect the losses
        # from all of them if there is a nested list
        if isinstance(outputs[0], list):
            outputs = itertools.chain(*outputs)

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / train", avg_losses["loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "loss":
                continue
            # Log the other losses
            self.log(loss_key + " / train", avg_losses[loss_key], sync_dist=True)

        # TODO: condition to verify
        # if avg_losses["loss"] < self.loss_threshold_until_certify:
        if self.ready_to_certify():
            print("Loss is below threshold, attempting certification")
            s_ub, s_lb = self.dynamics_model.state_limits
            normalization_A = torch.diag(self.k * 2 / (s_ub - s_lb))
            normalization_b = ((s_ub * (-self.k)) - (s_lb * self.k)) / (s_ub - s_lb)
            normalization_layer = nn.Linear(
                in_features=normalization_A.shape[1],
                out_features=normalization_A.shape[0],
                bias=True,
            )
            normalization_layer.weight = nn.Parameter(
                torch.FloatTensor(normalization_A)
            )
            normalization_layer.bias = nn.Parameter(torch.FloatTensor(normalization_b))

            v_nn = self.V_nn
            v_nn_with_normalization = nn.Sequential(normalization_layer, *v_nn)
            v_nn_merged = merge_adjacent_linear_layers(v_nn_with_normalization)

            from NCBCV.Cases import Darboux as Darboux_NCBCV
            from NCBCV.Cases import LinearSat as LinearSat_NCBCV
            from NCBCV.Cases import ObsAvoid as ObsAvoid_NCBCV
            from NCBCV.Cases import HighO as HighO_NCBCV

            if self.dynamics_model.__class__.__name__ == "Darboux":
                case = Darboux_NCBCV()
            elif self.dynamics_model.__class__.__name__ == "LinearSatellite":
                case = LinearSat_NCBCV()
            elif self.dynamics_model.__class__.__name__ == "ObsAvoid":
                case = ObsAvoid_NCBCV()
            elif self.dynamics_model.__class__.__name__ == "HighO":
                case = HighO_NCBCV()

            # case = NCBCV.Cases.LinearSat()
            hdlayers = []
            for layer in range(self.cbf_hidden_layers):
                hdlayers.append(("relu", self.cbf_hidden_size))
            architecture = (
                [("linear", self.dynamics_model.n_dims)] + hdlayers + [("linear", 1)]
            )
            model = NNet(architecture).to(self.device)
            trained_state_dict = v_nn_merged.state_dict()
            trained_state_dict = {
                f"layers.{key}": value for key, value in trained_state_dict.items()
            }
            model.load_state_dict(trained_state_dict, strict=True)

            safe_sample = self.dynamics_model.sample_safe(1)
            unsafe_sample = self.dynamics_model.sample_unsafe(1)
            while not all(self.dynamics_model.safe_mask(safe_sample)):
                safe_sample = self.dynamics_model.sample_safe(1)
            while not all(self.dynamics_model.unsafe_mask(unsafe_sample)):
                unsafe_sample = self.dynamics_model.sample_unsafe(1)
            print("Setup finished, starting certification...")
            if (model(safe_sample) * model(unsafe_sample)).squeeze().item() > 0:
                print("Safe sample: {}".format(safe_sample[0]))
                print(
                    "Safe sample output: {}".format(model(safe_sample).squeeze().item())
                )
                print("Unsafe sample: {}".format(unsafe_sample[0]))
                print(
                    "Unsafe sample output: {}".format(
                        model(unsafe_sample).squeeze().item()
                    )
                )
                print("Same sign - Verification failed!")
                self.CEs.append((self.current_epoch, safe_sample[0].detach().cpu().numpy()))
                self.CEs.append((self.current_epoch, unsafe_sample[0].detach().cpu().numpy()))
            else:
                spt = safe_sample.unsqueeze(0).to(self.device)
                uspt = unsafe_sample.unsqueeze(0).to(self.device)
                # Search Verification and output Counter Example
                # Search_prog = SearchVerifierMT(model, case)
                Search_prog = SearchVerifier(model, case)
                veri_flag, ce, info = Search_prog.SV_CE(spt, uspt)
                self.num_boundary_seg = info["num_boundary_seg"]
                if veri_flag:
                    print("Verification successful!")
                    self.early_stop = True
                    self.early_stop_epoch = self.current_epoch
                else:
                    print("Verification failed!")
                    print("Counter example:", ce)
                    if type(ce) == list:
                        for c in ce:
                            self.CEs.append((self.current_epoch, c))
                    else:
                        self.CEs.append((self.current_epoch, ce))

    def validation_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask)
        )
        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(x, goal_mask, safe_mask, unsafe_mask)
            )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        # Also compute the accuracy associated with each loss
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask, accuracy=True)
        )
        if self.current_epoch > self.learn_shape_epochs:
            component_losses.update(
                self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, accuracy=True)
            )

        batch_dict = {"val_loss": total_loss, **component_losses}

        return batch_dict

    def validation_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / val", avg_losses["val_loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "val_loss":
                continue
            # Log the other losses
            self.log(loss_key + " / val", avg_losses[loss_key], sync_dist=True)

        # **Now entering spicetacular automation zone**
        # We automatically run experiments every few epochs

        # Only plot every 5 epochs
        if self.current_epoch % 5 != 0:
            return

        self.experiment_suite.run_all_and_log_plots(
            self, self.logger, self.current_epoch
        )

    @pl.core.decorators.auto_move_data
    def simulator_fn(
        self,
        x_init: torch.Tensor,
        num_steps: int,
        relaxation_penalty: Optional[float] = None,
    ):
        # Choose parameters randomly
        random_scenario = {}
        for param_name in self.scenarios[0].keys():
            param_max = max([s[param_name] for s in self.scenarios])
            param_min = min([s[param_name] for s in self.scenarios])
            random_scenario[param_name] = random.uniform(param_min, param_max)

        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            self.u,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
            params=random_scenario,
        )

    def configure_optimizers(self):
        clbf_params = list(self.V_nn.parameters())

        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "clbf"}

        return [clbf_opt]
