from Verifier.Verification import *
from collections import deque
from Modules.Function import *
from SearchVerifier import SearchVerifier
from Scripts.Search_MT import *
import multiprocessing
import time
import os
import cProfile
import pstats
import io

class SearchVerifierMTP(SearchVerifier):
    def __init__(self, model, case) -> None:
        super().__init__(model, case)
        self.manager = multiprocessing.Manager()
        self.num_workers = multiprocessing.cpu_count()
        self.ce = self.manager.list()
        self.task_queue = self.manager.Queue()
        self.output_queue = self.manager.Queue()
        self.boundary_dict = self.manager.dict()
        self.boundary_list = self.manager.list()
        self.pair_wise_hinge = self.manager.list()
        self.visited = self.manager.dict()
        self.verifier = Verifier(model, case, unstable_neurons_set=[], pair_wise_hinge=[], ho_hinge=[])
        self.veri_flag = self.manager.Value('b', True)
        
    def worker(self):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            while True:
                if self.task_queue.empty():
                    self.task_queue.put(None)
                    break
                current_set = self.task_queue.get(timeout=0.001)
                previous_set = None
                hashable_d = tuple(sorted((k, tuple(v)) for k, v in current_set.items()))
                if hashable_d in self.visited:
                    continue
                self.visited[hashable_d] = True
                if not self.veri_flag.value:
                    print('Verification failed!')
                    print('Segment counter example', self.ce)
                    break
                # print(f"Processing: {current_set}")
                if current_set in self.boundary_list:
                    continue
                res = solver_lp(self.model, current_set, SSpace=self.case.SSpace)

                if res.is_success():
                    self.output_queue.put(current_set)
                    hashable_d = tuple(sorted((k, tuple(v)) for k, v in current_set.items()))
                    if hashable_d not in self.boundary_dict:
                        self.boundary_dict[hashable_d] = True
                        self.boundary_list.append(current_set)
                        if previous_set is not None:
                            pair_wise_hinge.append([previous_set, current_set])
                        previous_set = current_set

                        unstable_neighbours = self.Filter_S_neighbour(current_set)
                        unstable_neighbours_S = self.Possible_S(current_set, unstable_neighbours)

                        for item in unstable_neighbours_S:
                            hashable_u = tuple(sorted((k, tuple(v)) for k, v in item.items()))
                            if hashable_u not in self.boundary_dict:
                                self.task_queue.put(item)
                    
                    
                    veri_flag_container, ce_item = self.verifier.seg_verification([current_set], reverse_flag = self.case.reverse_flag)
                    self.veri_flag.value = veri_flag_container
                    if ce_item is not None:
                        self.ce.append(ce_item)
                    if not self.veri_flag or len(self.ce) > 0:
                        print('Verification failed!')
                        print('Segment counter example', self.ce)
                        break
                    
                    
            profiler.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            ps.print_stats()
        except Exception as e:
            # print(f"Worker error: {e}")
            return
        finally:
            # print("Worker terminated.")
            return

    def BFS_parallel_with_verifier(self, root_node):
        manager = multiprocessing.Manager()
        veri_flag = manager.Value('b', True)
        profiler = cProfile.Profile()
        profiler.enable()

        
        # num_workers = 1
        # for _ in range(num_workers):
        #     task_queue.put(None)
            
        # init_queue, pair_wise_hinge = self.BFS(root_node, termination=2*num_workers)
        self.task_queue.put(root_node)
        self.output_queue.put(root_node)
        
        # workers = [multiprocessing.Process(target=self.worker, args=(veri_flag, ce, task_queue, output_queue, boundary_dict, boundary_list, pair_wise_hinge)) for _ in range(num_workers)]
        workers = [multiprocessing.Process(target=self.worker) for _ in range(self.num_workers)]

        for w in workers:
            w.start()
            
        if self.task_queue.qsize() == 0:
            self.task_queue.put(None)

        for w in workers:
            if self.veri_flag.value:
                w.join()
        
        profiler.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        
        print(self.veri_flag.value)
        if not veri_flag.value or len(self.ce) > 0:
            print('Verification failed!')
            print('Segment counter example output', ce)
            return False, self.ce, list(self.boundary_list), list(pair_wise_hinge)
        else:
            print(f"Boundary list size: {len(self.boundary_list)}, Pair-wise hinge size: {len(self.pair_wise_hinge)}")
            return self.veri_flag.value, self.ce, list(self.boundary_list), list(self.pair_wise_hinge)
    
    
    def suff_check_hinge(self, unstable_neurons_set):
        prob_S_list = []
        if self.case.is_fx_linear:
            for S in unstable_neurons_set:
                veri_check = veri_seg_FG_wo_U(self.model, self.case, S)
                res_is_success, res_x, res_cost = veri_check.min_Lf(reverse_flag=self.case.reverse_flag)
                if res_cost < 0:
                    prob_S_list.append(S)
        return prob_S_list
    #     prob_S_list = []
    #     for S in unstable_neurons_set:
    #         self.verifier.seg_verifier._update_linear_exp(S)
    #         res_is_success, res_x, res_cost = self.verifier.seg_verifier.min_Lf(reverse_flag=self.case.reverse_flag)
    #         if res_cost < 0:
    #             prob_S_list.append(S)
    #     if self.case.is_gx_linear and not self.case.is_u_cons:
    #         for S in unstable_neurons_set:
    #             W_B, r_B, W_o, r_o = LinearExp(self.model, S)
    #             index_o = len(S.keys())-1
    #             Lgb = np.matmul(W_o[index_o], self.case.g_x(0))
    #             if np.equal(Lgb, np.zeros(self.case.CTRLDIM)).any():
    #                 prob_S_list.append(S)
    #     elif self.case.is_gx_linear and self.case.is_u_cons_interval:
    #         for S in unstable_neurons_set:
    #             veri_flag, veri_res_x = self.verifier.seg_verifier.min_NLf_interval(S, reverse_flag=self.case.reverse_flag)
    #             if not veri_flag:
    #                 prob_S_list.append(S)
    #     else:
    #         for S in unstable_neurons_set:
    #             veri_flag, veri_res_x = self.verifier.seg_verifier.min_NLfg(S, reverse_flag=self.case.reverse_flag)
    #             if not veri_flag:
    #                 prob_S_list.append(S)
    #     return prob_S_list
            
    def SV_CE(self, spt, uspt):
        # Search Verification and output Counter Example
        time_start = time.time()
        self.Specify_point(spt, uspt)
        result = self.BFS_parallel_with_verifier(self.S_init[0])
        if result:
            veri_flag, ce, unstable_neurons_set, pair_wise_hinge = result
            print('Verification flag:', veri_flag)
            print('Num boundary segments:', len(unstable_neurons_set))
        seg_search_time = time.time() - time_start
        print('Seg Search and Verification time:', seg_search_time)
        print('Num boundar seg is', len(unstable_neurons_set))
        if not self.veri_flag:
            return False, ce
        
        prob_S_checklist = self.suff_check_hinge(unstable_neurons_set)
        if len(prob_S_checklist) > 0:
            print('Num prob_S_checklist is', len(prob_S_checklist))
        else:
            print('No prob_S_checklist')
            return True, None
        o2_hinge = self.hinge_search_seg_comb(prob_S_checklist, pair_wise_hinge, n=2)
        if len(prob_S_checklist) >= len(pair_wise_hinge)/2:
            veri_flag, ce = self.verifier.hinge_verification(o2_hinge, reverse_flag=self.case.reverse_flag)
            if not veri_flag:
                return False, ce
        
        hinge2_search_time = time.time() - time_start - seg_search_time
        print('O2 Hinge Search time:', hinge2_search_time)
        print('Num O2 hinge is', len(o2_hinge))
        veri_flag, ce = self.verifier.hinge_verification(o2_hinge, reverse_flag=self.case.reverse_flag)
        hinge2_verification_time = time.time() - time_start - hinge2_search_time
        print('Pair-wise Hinge Verification time:', hinge2_verification_time)
        if not veri_flag:
            return False, ce
        
        o3_hinge = self.hinge_search_3seg(prob_S_checklist, o2_hinge)
        hinge3_search_time = time.time() - time_start - hinge2_verification_time
        print('O3 Hinge Search time:', hinge3_search_time)
        print('Num O3 hinge is', len(o3_hinge))
        veri_flag, ce = self.verifier.hinge_verification(o3_hinge, reverse_flag=self.case.reverse_flag)
        hinge3_verification_time = time.time() - time_start - hinge3_search_time
        print('Order-3 Hinge Verification time:', hinge3_verification_time)
        if not veri_flag:
            return False, ce
    
        ho_hinge = self.hinge_search(unstable_neurons_set, pair_wise_hinge)
        ho_hinge_search_time = time.time() - time_start - hinge3_verification_time
        print('HO Hinge Search time:', ho_hinge_search_time)
        veri_flag, ce = self.verifier.hinge_verification(ho_hinge, reverse_flag=self.case.reverse_flag)
        HOhinge_verification_time = time.time() - time_start - ho_hinge_search_time
        print('HO Hinge Verification time:', HOhinge_verification_time)
        if not veri_flag:
            return False, ce
        
        return True, None
    
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
    Search_prog = SearchVerifierMTP(model, case)
    veri_flag, ce = Search_prog.SV_CE(spt, uspt)
    if veri_flag:
        print('Verification successful!')
    else:
        print('Verification failed!')
        print('Counter example:', ce)
        