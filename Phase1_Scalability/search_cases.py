from Scripts.Search import *
import Scripts.PARA as PARA
from Cases.Darboux import Darboux
def test_darboux():
    case = Darboux()

    architecture = [('linear', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/darboux_1_32.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # case = PARA.CASES[0]
    Search_prog = Search(model)
    # (0.5, 1.5), (0, -1)
    Search_prog.Specify_point(torch.tensor([[[0.5, 1.5]]]), torch.tensor([[[-1, 0]]]))
    # print(Search.S_init)

    # Search.Filter_S_neighbour(Search.S_init[0])
    # Possible_S = Search.Possible_S(Search.S_init[0], Search.Filter_S_neighbour(Search.S_init[0]))
    # print(Search.Filter_S_neighbour(Search.S_init[0]))
    unstable_neurons_set = Search_prog.BFS(Search_prog.S_init[0])
    # unstable_neurons_set = Search.BFS(Possible_S)
    print(unstable_neurons_set)
    print(len(unstable_neurons_set))


if __name__ == "__main__":
    test_darboux()