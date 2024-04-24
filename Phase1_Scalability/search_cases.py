from Cases.ObsAvoid import ObsAvoid
from Scripts.Search import *
import Scripts.PARA as PARA
from Cases.Darboux import Darboux
import time

def test_darboux_single():
    case = Darboux()

    architecture = [('linear', 2), ('relu', 1024), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/models/darboux_1_1024.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # record time
    start = time.time()
    # case = PARA.CASES[0]
    Search_prog = Search(model)
    # (0.5, 1.5), (0, -1)
    Search_prog.Specify_point(torch.tensor([[[0.5, 1.5]]]), torch.tensor([[[-1, 0.1]]]))
    # print(Search.S_init)
    
    # Search.Filter_S_neighbour(Search.S_init[0])
    # Possible_S = Search.Possible_S(Search.S_init[0], Search.Filter_S_neighbour(Search.S_init[0]))
    # print(Search.Filter_S_neighbour(Search.S_init[0]))
    unstable_neurons_set = Search_prog.BFS(Search_prog.S_init[0])
    # unstable_neurons_set = Search.BFS(Possible_S)

    # compute searching time
    end = time.time()
    

    # print(unstable_neurons_set)
    print(len(unstable_neurons_set))
    print("Time:", end - start)

def test_darboux():
    case = Darboux()

    architecture = [('linear', 2), ('relu', 512), ('relu', 512), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/models/darboux_2_512.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # record time
    start = time.time()
    # case = PARA.CASES[0]
    Search_prog = Search(model)
    # (0.5, 1.5), (0, -1)
    Search_prog.Specify_point(torch.tensor([[[0.5, 1.5]]]), torch.tensor([[[-1, 0.1]]]))
    # print(Search.S_init)
    
    # Search.Filter_S_neighbour(Search.S_init[0])
    # Possible_S = Search.Possible_S(Search.S_init[0], Search.Filter_S_neighbour(Search.S_init[0]))
    # print(Search.Filter_S_neighbour(Search.S_init[0]))
    unstable_neurons_set = Search_prog.BFS(Search_prog.S_init[0])
    # unstable_neurons_set = Search.BFS(Possible_S)

    # compute searching time
    end = time.time()
    

    # print(unstable_neurons_set)
    print(len(unstable_neurons_set))
    print("Time:", end - start)

def test_obs():
    # case = ObsAvoid()
    architecture = [('linear', 3), ('relu', 32), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("Phase1_Scalability/models/obs_2_32.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # record time
    start = time.time()
    # case = PARA.CASES[0]
    Search_prog = Search(model)
    # (0.5, 1.5), (0, -1)
    spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
    Search_prog.Specify_point(spt, uspt)
    unstable_neurons_set = Search_prog.BFS(Search_prog.S_init[0])
    end = time.time()
    # print(unstable_neurons_set)
    print(len(unstable_neurons_set))
    print("Time:", end - start)

def test_obs_single():
    # case = ObsAvoid()
    architecture = [('linear', 3), ('relu', 64), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("Phase1_Scalability/models/obs_1_64.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # record time
    start = time.time()
    # case = PARA.CASES[0]
    Search_prog = Search(model)
    # (0.5, 1.5), (0, -1)
    spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
    Search_prog.Specify_point(spt, uspt)
    unstable_neurons_set = Search_prog.BFS(Search_prog.S_init[0])
    end = time.time()
    # print(unstable_neurons_set)
    print(len(unstable_neurons_set))
    print("Time:", end - start)

def test_sate():
    case = Darboux()

    architecture = [('linear', 6), ('relu', 32), ('relu', 32), ('linear', 32), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/models/satellitev1_2_32.pt")
    key_map = {
        'V_nn.input_linear.weight': 'layers.0.weight',
        'V_nn.input_linear.bias': 'layers.0.bias',
        'V_nn.layer_0_linear.weight': 'layers.2.weight',
        'V_nn.layer_0_linear.bias': 'layers.2.bias',
        'V_nn.layer_1_linear.weight': 'layers.4.weight', # Adjust this line if the layers don't match
        'V_nn.layer_1_linear.bias': 'layers.4.bias',     # Adjust this line if the layers don't match
        'V_nn.output_linear.weight': 'layers.6.weight', # Adjust this line if the layers don't match
        'V_nn.output_linear.bias': 'layers.6.bias',     # Adjust this line if the layers don't match
    }

    # Remap the state dict keys according to the defined mapping
    remapped_state_dict = {key_map[key]: value for key, value in trained_state_dict.items() if key in key_map}

    
    # trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(remapped_state_dict, strict=True)
    # record time
    start = time.time()
    # case = PARA.CASES[0]
    Search_prog = Search(model)
    # (0.5, 1.5), (0, -1)
    spt = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    Search_prog.Specify_point(spt, uspt)
    # print(Search.S_init)
    
    # Search.Filter_S_neighbour(Search.S_init[0])
    # Possible_S = Search.Possible_S(Search.S_init[0], Search.Filter_S_neighbour(Search.S_init[0]))
    # print(Search.Filter_S_neighbour(Search.S_init[0]))
    unstable_neurons_set = Search_prog.BFS(Search_prog.S_init[0])
    # unstable_neurons_set = Search.BFS(Possible_S)

    # compute searching time
    end = time.time()
    

    # print(unstable_neurons_set)
    print(len(unstable_neurons_set))
    print("Time:", end - start)

if __name__ == "__main__":
    # test_darboux()
    # test_darboux_single()
    # test_obs()
    test_obs_single()