from mimetypes import init
from re import search
from socket import timeout
import sys, os
import multiprocessing
from collections import deque
import time
from numpy import empty

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
from Scripts.Search import *
from Cases.ObsAvoid import ObsAvoid

from RaycastInitSearch import *

class SearchMTRaycast(Search):
    def __init__(self, model, case=None) -> None:
        super().__init__(model, case)
        self.model = model
        self.case = case
        self.NStatus = NetworkStatus(model)
        self.verbose = False
    
    def worker(self, task_queue, output_queue, boundary_dict, boundary_list, pair_wise_hinge):
        try:
            while True:
                current_set = task_queue.get()
                # if current_set is None:
                #     print("Worker received termination signal.")
                #     break
                if task_queue.empty():
                    print("Worker received termination signal.")
                    break
                # print(f"Processing: {current_set}")
                res = solver_lp(self.model, current_set, SSpace=self.case.SSpace)
                if res.is_success():
                    output_queue.put(current_set)
                    hashable_d = tuple(sorted((k, tuple(v)) for k, v in current_set.items()))
                    if hashable_d not in boundary_dict:
                        boundary_dict.append(hashable_d)
                        boundary_list.append(current_set)
                        # print(f"Added to boundary list: {current_set}")

                        unstable_neighbours = self.Filter_S_neighbour(current_set)
                        unstable_neighbours_S = self.Possible_S(current_set, unstable_neighbours)
                        # print(f"Neighbours count: {len(unstable_neighbours_S)}")

                        for item in unstable_neighbours_S:
                            hashable_u = tuple(sorted((k, tuple(v)) for k, v in item.items()))
                            if hashable_u not in boundary_dict:
                                task_queue.put(item)
                                # print(f"Enqueued: {item}")
        except Exception as e:
            print(f"Worker error: {e}")
            return
        finally:
            print("Worker terminated.")
            return

    def BFS_parallel(self):
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        output_queue = manager.Queue()
        boundary_dict = manager.list()
        boundary_list = manager.list()
        pair_wise_hinge = manager.list()

        num_workers = multiprocessing.cpu_count()
        # num_workers = 1
        # for _ in range(num_workers):
        #     task_queue.put(None)
        Origin = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
        Direction = [np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]
        RCSearch = RaycastSearch(model, case, same_origin=True, same_direction=False)
        RCSearch.num_rays = 5
        RCSearch.raycast(Origin, Direction)
        init_act_set = RCSearch.list_activation_intersections
        init_queue = []
        pair_wise_hinge = []
        init_queue = init_act_set
        # for root_node in init_act_set:
        #     init_queue_item, pair_wise_hinge_item = self.BFS(root_node, termination=2*num_workers)
        #     init_queue += init_queue_item
        #     pair_wise_hinge += pair_wise_hinge_item
        
        # init_queue, pair_wise_hinge = self.BFS(root_node, termination=2*num_workers)
        for item in init_queue:
            task_queue.put(item)
        
        workers = [multiprocessing.Process(target=self.worker, args=(task_queue, output_queue, boundary_dict, boundary_list, pair_wise_hinge)) for _ in range(num_workers)]

        for w in workers:
            w.start()

        for w in workers:
            w.join()
            
        for w in workers:
            w.terminate()
        
        print(f"Boundary list size: {len(boundary_list)}, Pair-wise hinge size: {len(pair_wise_hinge)}")
        return list(boundary_list), list(pair_wise_hinge)
        
if __name__ == "__main__":
    # architecture = [('linear', 3), ('relu', 64), ('linear', 1)]
    # model = NNet(architecture)
    
    # trained_state_dict = torch.load("models/obs_1_64.pt")
    # trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    # model.load_state_dict(trained_state_dict, strict=True)
    
    # TODO: debug multiple thread 
    # TODO: rewrite multiple thread part and make it working stably
    from Cases.LinearSatellite import LinearSat
    from Modules.NNet import NeuralNetwork as NNet
    
    case = LinearSat()
    n = 16
    architecture = [('linear', 6), ('relu', n), ('relu', n), ('relu', n), ('linear', 1)]
    model = NNet(architecture)
    # trained_state_dict = torch.load(f"Phase2_Verification/models/linear_satellite_hidden_32_epoch_50_reg_0.05.pt")
    trained_state_dict = torch.load(f"models/linear_satellite_layer_3_hidden_16_epoch_50_reg_0.pt")
    model.load_state_dict_from_sequential(trained_state_dict)
    
    
    # case = PARA.CASES[0]
    # case = ObsAvoid()
    # Search = Search(model)
    start_time = time.time()
    Search_prog = SearchMTRaycast(model, case)
    
    # (0.5, 1.5), (0, -1)
    # spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
    # uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
    # Search_prog.Specify_point(spt, uspt)
    # print(Search.S_init)

    # Search.Filter_S_neighbour(Search.S_init[0])
    # Possible_S = Search.Possible_S(Search.S_init[0], Search.Filter_S_neighbour(Search.S_init[0]))
    # print(Search.Filter_S_neighbour(Search.S_init[0]))
    # unstable_neurons_set, pair_wise_hinge = Search.BFS(Search.S_init[0])
    unstable_neurons_set, pair_wise_hinge = Search_prog.BFS_parallel()
    
    search_time = time.time() - start_time
    print('Search time:', search_time)
    
    # unstable_neurons_set = Search.BFS(Possible_S)
    # print(unstable_neurons_set)
    print(len(unstable_neurons_set))
    print(len(pair_wise_hinge))
    print(f"Boundary list size: {len(unstable_neurons_set)}, Pair-wise hinge size: {len(pair_wise_hinge)}")
    # ho_hinge = Search.hinge_search(unstable_neurons_set, pair_wise_hinge)
    # print(len(ho_hinge))
