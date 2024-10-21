from re import search
from socket import timeout
import sys, os
import multiprocessing
from collections import deque
import time
from numpy import empty

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
from Scripts.Search import *
from RaycastInitSearch import RaycastSearch
from Cases.ObsAvoid import ObsAvoid

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
                        print(f"Neighbours count: {len(unstable_neighbours_S)}")

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

    def BFS_parallel(self, root_node):
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
            
        # init_queue, pair_wise_hinge = self.BFS(root_node, termination=2*num_workers)
        for item in root_node:
            task_queue.put(item)
        
        workers = [multiprocessing.Process(target=self.worker, args=(task_queue[i], output_queue, boundary_dict, boundary_list, pair_wise_hinge)) for i in range(num_workers)]

        for w in workers:
            w.start()

        # for w in workers:
        #     w.join()
            
        # for w in workers:
        #     w.terminate()
        
        print(f"Boundary list size: {len(boundary_list)}, Pair-wise hinge size: {len(pair_wise_hinge)}")
        return list(boundary_list), list(pair_wise_hinge)
        
if __name__ == "__main__":
    
    from Cases.LinearSatellite import LinearSat
    from Modules.NNet import NeuralNetwork as NNet
    
    l = 1
    n = 128
    case = ObsAvoid()
    hdlayers = []
    for layer in range(l):
        hdlayers.append(('relu', n))
    architecture = [('linear', 3)] + hdlayers + [('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load(f"models/obs_{l}_{n}.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    
    start_time = time.time()
    Search_prog = SearchMTRaycast(model, case)
    Origin = [np.array([0.0, 0.0, 0.0])]
    RCSearch = RaycastSearch(model, case, same_origin=True, same_direction=False)
    RCSearch.num_rays = 8
    RCSearch.raycast(Origin)
    init_act_set = RCSearch.list_activation_intersections
    unstable_neurons_set, pair_wise_hinge = Search_prog.BFS_parallel(init_act_set[:32])
    
    search_time = time.time() - start_time
    print('Search time:', search_time)
    print(f"Boundary list size: {len(unstable_neurons_set)}, Pair-wise hinge size: {len(pair_wise_hinge)}")