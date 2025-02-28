from typing import Tuple, Optional, List
import torch
from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList

class HighO(ControlAffineSystem):

    # Number of states and controls
    N_DIMS = 8
    N_CONTROLS = 1

    # State indices
    X0 = 0
    X1 = 1
    X2 = 2
    X3 = 3
    X4 = 4
    X5 = 5
    X6 = 6
    X7 = 7

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
        use_l1_norm: bool = False,
    ):
        """
        Initialize the HighO system.

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
        return HighO.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return HighO.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this system.
        """
        upper_limit = torch.ones(self.n_dims) * 2.0
        lower_limit = torch.ones(self.n_dims) * -2.0
        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control limits for this system.
        """
        upper_limit = torch.tensor([0.0])
        lower_limit = torch.tensor([0.0])
        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task.

        args:
            x: a tensor of points in the state space.
        """
        hx = torch.sum((x + 2) ** 2, dim=1) - 3
        safe_mask = hx >= 0
        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task.

        args:
            x: a tensor of points in the state space.
        """
        hx = torch.sum((x + 2) ** 2, dim=1) - 3
        unsafe_mask = hx < 0
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

        f[:, HighO.X0, 0] = x[:, HighO.X1]
        f[:, HighO.X1, 0] = x[:, HighO.X2]
        f[:, HighO.X2, 0] = x[:, HighO.X3]
        f[:, HighO.X3, 0] = x[:, HighO.X4]
        f[:, HighO.X4, 0] = x[:, HighO.X5]
        f[:, HighO.X5, 0] = x[:, HighO.X6]
        f[:, HighO.X6, 0] = x[:, HighO.X7]
        f[:, HighO.X7, 0] = (-20 * x[:, HighO.X7]
                            - 170 * x[:, HighO.X6]
                            - 800 * x[:, HighO.X5]
                            - 2273 * x[:, HighO.X4]
                            - 3980 * x[:, HighO.X3]
                            - 4180 * x[:, HighO.X2]
                            - 2400 * x[:, HighO.X1]
                            - 576 * x[:, HighO.X0])

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

        return g

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
        return torch.zeros(x.shape[0], self.n_controls).to(x.device)
