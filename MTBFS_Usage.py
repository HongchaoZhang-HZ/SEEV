from ast import arg
from Verifier.Verification import *
from collections import deque
from Modules.Function import *
from SearchVerifier import SearchVerifier
from Scripts.Search_MT import *
import multiprocessing
import time
import os

class SearchMT(Search):
    def __init__(self, model, case=None) -> None:
        super().__init__(model, case)
        self.model = model
        self.case = case
        self.NStatus = NetworkStatus(model)
        self.verbose = False

        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.output_queue = self.manager.Queue()
        self.boundary_dict = self.manager.dict()
        self.boundary_list = self.manager.list()
        self.pair_wise_hinge = self.manager.list()
        self.num_workers = multiprocessing.cpu_count()

    def worker(self):
        while True:
            print('num of tasks:', self.task_queue.qsize())
            current_set = self.task_queue.get()
            if self.task_queue.empty():
                break
            if current_set is None:
                print(f"Worker {os.getpid()} received termination signal.")
                break
            try:
                res = solver_lp(self.model, current_set, SSpace=self.case.SSpace)
                if res.is_success():
                    self.output_queue.put(current_set)
                    hashable_d = tuple(sorted((k, tuple(v)) for k, v in current_set.items()))
                    if hashable_d not in self.boundary_dict:
                        self.boundary_dict[hashable_d] = True
                        self.boundary_list.append(current_set)

                        unstable_neighbours = self.Filter_S_neighbour(current_set)
                        unstable_neighbours_S = self.Possible_S(current_set, unstable_neighbours)

                        for item in unstable_neighbours_S:
                            hashable_u = tuple(sorted((k, tuple(v)) for k, v in item.items()))
                            if hashable_u not in self.boundary_dict:
                                self.task_queue.put(item)
            except Exception as e:
                print(f"Worker error: {e}")

    def BFS_parallel(self, root_node):
        print("Initializing BFS_parallel...")
        init_queue, self.pair_wise_hinge = self.BFS(root_node, termination=2*self.num_workers)
        for item in init_queue:
            self.task_queue.put(item)
        print(f"Initial task queue size: {self.task_queue.qsize()}")

        workers = [multiprocessing.Process(target=self.worker) for _ in range(self.num_workers)]

        for w in workers:
            w.start()
            
        # Wait for all workers to finish
        for w in workers:
            w.join()

        # Add sentinel values to stop the workers
        if self.task_queue.empty():
            print("Worker received termination signal.")
            self.task_queue.put(None)
        
        for w in workers:
            w.terminate()

        print(f"Boundary list size: {len(self.boundary_list)}, Pair-wise hinge size: {len(self.pair_wise_hinge)}")
        return list(self.boundary_list), list(self.pair_wise_hinge)

    
if __name__ == "__main__":
    # CBF Verification
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
    
    spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
    # Search Verification and output Counter Example
    start_time = time.time()
    Search_prog = SearchMT(model, case)
    Search_prog.Specify_point(spt, uspt)
    Search_prog.BFS_parallel(Search_prog.S_init[0])
    end_time = time.time() - start_time
    print('Time taken:', end_time)
    # veri_flag, ce = Search_prog.SV_CE(spt, uspt)
    # if veri_flag:
    #     print('Verification successful!')
    # else:
    #     print('Verification failed!')
    #     print('Counter example:', ce)
        