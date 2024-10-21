from ast import arg
from Verifier.Verification import *
from collections import deque
from Modules.Function import *
from SearchVerifier import SearchVerifier
from MTBFS_UsageProfile import SearchMT
import multiprocessing
import time
import os
from RaycastInitSearch import *
from Cases.ObsAvoid import ObsAvoid
import os
import cProfile
import pstats
import io

class SearchMTRC(SearchMT):
    def __init__(self, model, case=None, Origin=None, Direction=None, same_origin=True, same_direction=False) -> None:
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
        self.visited = self.manager.dict()
        
        self.Origin = Origin
        self.Direction = Direction
        
        self.RCS = RaycastSearch(model, case, same_origin=True, same_direction=False)
        self.RCS.num_rays = 4
        self.RCS.raycast(self.Origin, self.Direction)

    def BFS_parallel(self):
        profiler = cProfile.Profile()
        profiler.enable()
        print("Initializing BFS_parallel...")
        # init_queue, self.pair_wise_hinge = self.BFS(root_node, termination=2*self.num_workers)
        # print(len(self.RCS.list_activation_intersections))
        for item in self.RCS.list_activation_intersections:
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
        
        profiler.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        # print(s.getvalue())

        print(f"Boundary list size: {len(self.boundary_list)}, Pair-wise hinge size: {len(self.pair_wise_hinge)}")
        return list(self.boundary_list), list(self.pair_wise_hinge)
    

def obs_rcMT():
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
    
    # Search Verification and output Counter Example
    Origin = [np.array([0.0, 0.0, 0.0])]
    Direction = [np.array([-1.0, 0.0, 0.0])]
    Search_prog = SearchMTRC(model, case, Origin, Direction)
    start_time = time.time()
    # Search_prog.Specify_point(spt, uspt)
    # Search_prog.RCS.raycast(Origin, Direction)
    Search_prog.BFS_parallel()
    
    end_time = time.time() - start_time
    print('Time taken:', end_time)
    
if __name__ == "__main__":
    # CBF Verification
    from Cases.LinearSatellite import LinearSat
    case = LinearSat()
    n = 8
    architecture = [('linear', 6), ('relu', n), ('relu', n), ('relu', n), ('relu', n), ('linear', 1)]
    model = NNet(architecture)
    # trained_state_dict = torch.load(f"Phase2_Verification/models/linear_satellite_hidden_32_epoch_50_reg_0.05.pt")
    trained_state_dict = torch.load(f"models/linear_satellite_layer_4_hidden_8_epoch_50_reg_0.1.pt")
    model.load_state_dict_from_sequential(trained_state_dict)

    
    # Search Verification and output Counter Example
    Origin = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    Direction = [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    Search_prog = SearchMTRC(model, case, Origin=Origin, Direction=Direction, same_origin=True, same_direction=False)
    start_time = time.time()
    # Search_prog.Specify_point(spt, uspt)
    # Search_prog.RCS.raycast(Origin, Direction)
    Search_prog.BFS_parallel()
    
    end_time = time.time() - start_time
    print('Time taken:', end_time)
    
    # num = 0
    # for item in Search_prog.boundary_list:
    #     hashable_d = tuple(sorted((k, tuple(v)) for k, v in item.items()))
    #     if hashable_d in Search_prog.boundary_dict:
    #         num += 1
    # print(f"Number of items in boundary list: {num}")
    
    # veri_flag, ce = Search_prog.SV_CE(spt, uspt)
    # if veri_flag:
    #     print('Verification successful!')
    # else:
    #     print('Verification failed!')
    #     print('Counter example:', ce)
        