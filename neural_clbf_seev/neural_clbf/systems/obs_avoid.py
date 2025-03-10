from typing import Tuple, Optional, List
from math import sin, cos

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class ObsAvoid(ControlAffineSystem):

    # Number of states and controls
    N_DIMS = 3
    N_CONTROLS = 1

    # State indices
    X = 0
    Y = 1
    PHI = 2

    # Control indices
    U = 0

    # Constant parameters
    VELOCITY = 1.0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
        use_l1_norm: bool = False,
    ):
        """
        Initialize the ObsAvoid system.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
            dt: the timestep to use for the simulation.
            controller_dt: the timestep for the LQR discretization. Defaults to dt.
            use_l1_norm: if True, use L1 norm for safety zones; otherwise, use L2.
        raises:
            ValueError if nominal_params are not valid for this system.
        """
        super().__init__(
            nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            scenarios=scenarios,
            use_linearized_controller=False,
        )
        self.use_l1_norm = use_l1_norm
        self.P = torch.eye(self.n_dims).float()

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid.

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise.
        """
        return True

    @property
    def n_dims(self) -> int:
        return ObsAvoid.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return ObsAvoid.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this system.
        """
        upper_limit = torch.tensor([2.0, 2.0, np.pi])
        lower_limit = torch.tensor([-2.0, -2.0, -np.pi])
        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control limits for this system.
        """
        upper_limit = torch.tensor([2.0])
        lower_limit = torch.tensor([-2.0])
        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task.

        args:
            x: a tensor of points in the state space.
        """
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : ObsAvoid.Y + 1].norm(dim=-1, p=order)
        safe_mask = distance >= 1.5
        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task.

        args:
            x: a tensor of points in the state space.
        """
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : ObsAvoid.Y + 1].norm(dim=-1, p=order)
        unsafe_mask = distance < 0.2
        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set.

        args:
            x: a tensor of points in the state space.
        """
        return self.safe_mask(x)

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state.
            params: a dictionary giving the parameter values for the system. If None, default to the nominal parameters used at initialization.
        returns:
            f: bs x self.n_dims x 1 tensor.
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1)).type_as(x)

        # Extract the needed parameters
        x_phi = x[:, ObsAvoid.PHI]

        f[:, ObsAvoid.X, 0] = ObsAvoid.VELOCITY * torch.sin(x_phi)
        f[:, ObsAvoid.Y, 0] = ObsAvoid.VELOCITY * torch.cos(x_phi)
        f[:, ObsAvoid.PHI, 0] = 0

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-dependent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state.
            params: a dictionary giving the parameter values for the system. If None, default to the nominal parameters used at initialization.
        returns:
            g: bs x self.n_dims x self.n_controls tensor.
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls)).type_as(x)

        g[:, ObsAvoid.PHI, ObsAvoid.U] = 1.0

        return g

    # def u_nominal(
    #     self, x: torch.Tensor, params: Optional[Scenario] = None
    # ) -> torch.Tensor:
    #     """Return the nominal control input.

    #     args:
    #         x: bs x self.n_dims tensor of state.
    #         params: a dictionary giving the parameter values for the system. If None, default to the nominal parameters used at initialization.
    #     returns:
    #         u: bs x self.n_controls tensor of control inputs.
    #     """
    #     return torch.zeros(x.shape[0], self.n_controls).to(x.device)

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """Return the nominal control input.

        args:
            x: bs x self.n_dims tensor of state.
            params: a dictionary giving the parameter values for the system. If None, default to the nominal parameters used at initialization.
        returns:
            u: bs x self.n_controls tensor of control inputs.
        """
        KP = 10.0
        position = x[:, :2]
        phi = x[:, 2]

        # Compute the distance vector from the obstacle
        distance_vector = position
        distance = torch.norm(distance_vector, dim=1, keepdim=True)

        # Compute the repulsive force if within the repulsion radius
        repulsion_force = torch.zeros_like(distance_vector)
        repulsion_force = KP * (1.0 / (distance + 1e-6)) * (distance_vector / (distance**3 + 1e-6))

        # Project the repulsive force onto the control input direction
        u = -torch.sum(
            repulsion_force * torch.stack([torch.sin(phi), torch.cos(phi)], dim=1),
            dim=1,
            keepdim=True,
        )

        u = torch.clamp(u, -self.control_limits[0], self.control_limits[0])

        return u
