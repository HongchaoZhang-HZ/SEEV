import torch
import torch.nn as nn
import numpy as np

from neural_clbf.systems import ControlAffineSystem


def normalize(
    dynamics_model: ControlAffineSystem, x: torch.Tensor, k: float = 1.0
) -> torch.Tensor:
    """Normalize the state input to [-k, k]

    args:
        dynamics_model: the dynamics model matching the provided states
        x: bs x self.dynamics_model.n_dims the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    x_max, x_min = dynamics_model.state_limits
    x_center = (x_max + x_min) / 2.0
    x_range = (x_max - x_min) / 2.0
    # Scale to get the input between (-k, k), centered at 0
    x_range = x_range / k
    # We shouldn't scale or offset any angle dimensions
    x_center[dynamics_model.angle_dims] = 0.0
    x_range[dynamics_model.angle_dims] = 1.0

    # Do the normalization
    return (x - x_center.type_as(x)) / x_range.type_as(x)


def normalize_with_angles(
    dynamics_model: ControlAffineSystem, x: torch.Tensor, k: float = 1.0
) -> torch.Tensor:
    """Normalize the input using the stored center point and range, and replace all
    angles with the sine and cosine of the angles

    args:
        dynamics_model: the dynamics model matching the provided states
        x: bs x self.dynamics_model.n_dims the points to normalize
        k: normalize non-angle dimensions to [-k, k]
    """
    # Scale and offset based on the center and range
    x = normalize(dynamics_model, x, k)

    # Replace all angles with their sine, and append cosine
    angle_dims = dynamics_model.angle_dims
    angles = x[:, angle_dims]
    x[:, angle_dims] = torch.sin(angles)
    x = torch.cat((x, torch.cos(angles)), dim=-1)

    return x

def merge_adjacent_linear_layers(model: nn.Sequential) -> nn.Sequential:
    done = False
    while not done:
        new_module_list = []
        done = True
        layer_idx = 0
        while layer_idx < len(model):
            layer = model[layer_idx]
            if isinstance(layer, nn.Linear):
                if layer_idx + 1 < len(model) and isinstance(
                    model[layer_idx + 1], nn.Linear
                ):

                    next_layer = model[layer_idx + 1]
                    A1 = layer.weight.data.detach().cpu().numpy()
                    b1 = layer.bias.data.detach().cpu().numpy()
                    A2 = next_layer.weight.data.detach().cpu().numpy()
                    b2 = next_layer.bias.data.detach().cpu().numpy()
                    A_new = A2 @ A1
                    b_new = (
                        A2 @ np.expand_dims(b1, 0).T + np.expand_dims(b2, 0).T
                    ).squeeze(axis=1)
                    new_layer = nn.Linear(A_new.shape[1], A_new.shape[0])
                    new_layer.weight.data = nn.Parameter(
                        torch.tensor(A_new, dtype=torch.float32)
                    )
                    new_layer.bias.data = nn.Parameter(
                        torch.tensor(b_new, dtype=torch.float32)
                    )
                    new_module_list.append(new_layer)
                    done = False
                    layer_idx += 2
                else:
                    new_module_list.append(layer)
                    layer_idx += 1
            else:
                new_module_list.append(layer)
                layer_idx += 1

        model = nn.Sequential(*new_module_list)

    return model