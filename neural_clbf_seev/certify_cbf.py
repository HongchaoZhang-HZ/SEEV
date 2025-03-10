import torch

from neural_clbf.systems import ObsAvoid
from neural_clbf.systems import LinearSatellite
from neural_clbf.systems import Darboux
from neural_clbf.systems import HighO

from EEV.Cases import LinearSat as LinearSat_EEV
from EEV.Cases import ObsAvoid as ObsAvoid_EEV
from EEV.Cases import Darboux as Darboux_EEV
from EEV.Cases import HighO as HighO_EEV

from EEV.SearchVerifierMT import SearchVerifierMT
from EEV.Modules.NNet import NeuralNetwork as NNet

def get_dynamics_model(system_name):
    simulation_dt = 0.01
    controller_period = 0.01
    if system_name == "linear_satellite":
        nominal_params = {
        "a": 500e3,
        "ux_target": 0.0,
        "uy_target": 0.0,
        "uz_target": 0.0,
        }
        scenarios = [
            nominal_params,
        ]

        dynamics_model = LinearSatellite(
            nominal_params,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=scenarios,
            use_l1_norm=False,
        )
    elif system_name == "obs_avoid":
        nominal_params = {}
        scenarios = [
            nominal_params,
        ]

        # Define the dynamics model
        dynamics_model = ObsAvoid(
            nominal_params,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=scenarios,
            use_l1_norm=False,
        )
    elif system_name == "darboux":
        nominal_params = {}
        scenarios = [
            nominal_params,
        ]

        # Define the dynamics model
        dynamics_model = Darboux(
            nominal_params,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=scenarios,
            use_l1_norm=False,
        )
    elif system_name == "high_o":
        nominal_params = {}
        scenarios = [
            nominal_params,
        ]

        # Define the dynamics model
        dynamics_model = HighO(
            nominal_params,
            dt=simulation_dt,
            controller_dt=controller_period,
            scenarios=scenarios,
            use_l1_norm=False,
        )
    else:
        raise ValueError(f"Unknown system name: {system_name}")
    
    return dynamics_model

def get_case(system_name):
    if system_name == "linear_satellite":
        case = LinearSat_EEV()
    elif system_name == "obs_avoid":
        case = ObsAvoid_EEV()
    elif system_name == "darboux":
        case = Darboux_EEV()
    elif system_name == "high_o":
        case = HighO_EEV()
    else:
        raise ValueError(f"Unknown system name: {system_name}")
    return case

def main(args):
    system_name = args.system_name
    cbf_hidden_layers = args.cbf_hidden_layers
    cbf_hidden_size = args.cbf_hidden_size
    model_path = args.model_path
    loaded = torch.load(model_path)

    dynamics_model = get_dynamics_model(system_name)
    case = get_case(system_name)
    hdlayers = []
    for layer in range(cbf_hidden_layers):
        hdlayers.append(("relu", cbf_hidden_size))
    architecture = [("linear", dynamics_model.N_DIMS)] + hdlayers + [("linear", 1)]
    model = NNet(architecture)

    try:
        trained_state_dict = loaded.state_dict()
    except AttributeError:
        trained_state_dict = loaded

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
    Search_prog = SearchVerifierMT(model, case)
    # Search_prog = SearchVerifier(model, case)
    import time
    start = time.time()
    veri_flag, ce, info = Search_prog.SV_CE(spt, uspt)
    end = time.time()
    print("Time taken:", end - start)
    from pprint import pprint
    pprint(info)
    num_boundary_seg = info["num_boundary_seg"]
    if veri_flag:
        print("Verification successful!")
    else:
        print("Verification failed!")
        print("Counter example:", ce)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_name", type=str, required=True)
    parser.add_argument("--cbf_hidden_layers", type=int, required=True)
    parser.add_argument("--cbf_hidden_size", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)