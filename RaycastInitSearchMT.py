from hmac import new
import multiprocessing.pool
from numpy import sign
from torch import ne
from Cases import ObsAvoid
import RaycastInitSearch
from Verifier.VeriUtil import *
from Scripts.Status import NeuronStatus, NetworkStatus
from RaycastInitSearch import *
import multiprocessing
import cProfile
from multiprocessing import Pool, Manager
import os
import cProfile
import pstats
import io
''' Raycast Initialial Search

'''



# Raycast Initialial Search:
class RaycastMT(RaycastSearch):
    def __init__(self, model, case, 
                 same_origin=None, same_direction=None):
        super().__init__(model, case, same_origin, same_direction)
        self.num_workers = multiprocessing.cpu_count()
        self.task_queue = Manager().Queue()
        self.list_intersection = Manager().list()
        self.list_rays = Manager().list()
        self.list_activation_intersections = Manager().list()
        
    def raycastworker(self):
        while True:
            if self.task_queue.empty():
                print("Worker received termination signal.")
                break
            current_args = self.task_queue.get()
            if current_args is None:
                print(f"Worker {os.getpid()} received termination signal.")
                break
            RCSearch = RaycastSearch(model, case, same_origin=True, same_direction=False)
            RCSearch.raycast(current_args)
            if len(RCSearch.list_intersection) >= 0:
                self.list_intersection.extend(RCSearch.list_intersection)
    
    def MTSearch(self, list_Origin:list=None, list_Direction:list=None):
        profiler = cProfile.Profile()
        profiler.enable()
        # print(f"Initial task queue size: {self.task_queue.qsize()}")
        # args_list = [(origin, direction) for origin in list_Origin for direction in list_Direction]
        for i in range(self.num_workers):
            self.task_queue.put((list_Origin[i], list_Direction[i]))
        for i in range(self.num_workers):
            self.task_queue.put(None)
        
        workers = [multiprocessing.Process(target=self.raycastworker) for _ in range(self.num_workers)]

        for w in workers:
            w.start()

        # Add sentinel values to stop the workers
        if self.task_queue.qsize() == 0:
            self.task_queue.put(None)

        # Wait for all workers to finish
        for w in workers:
            w.join()
            
        # with Manager() as manager:
        #     with Pool() as pool:
        #         results = pool.map(self.raycastworker, list_Origin)

        #     for ray, intersections, activations in results:
        #         self.list_rays.append(ray)
        #         self.list_intersection.extend(intersections)
        #         self.list_activation_intersections.extend(activations)
        
        profiler.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        return self.list_intersection, self.list_activation_intersections
    
if __name__ == "__main__":
    from Cases.LinearSatellite import LinearSat
    from Cases.ObsAvoid import ObsAvoid
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
    
    Origin = np.array([0.0, 0.0, 0.0])
    Origin_list = [Origin for _ in range(32)]
    Direction = np.array([-1.0, 0.0, 0.0])
    Direction_list = [Direction for _ in range(32)]
    # ray = Ray(model, case, Origin[0], Direction[0])
    # ray.rayspread()
    # print(len(ray.list_intersection))
    
    RCSearch = RaycastMT(model, case, same_origin=True, same_direction=False)
    RCSearch.num_rays = 10
    list_intersection, list_activation_intersections = RCSearch.MTSearch(Origin_list, Direction_list)
    print('Found boundary points', len(list_intersection))