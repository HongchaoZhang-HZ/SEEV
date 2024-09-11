from Verifier.Verification import *
from collections import deque
from Modules.Function import *
from SearchVerifier import SearchVerifier
from Scripts.Search_MT import *
class SearchVerifierMT(SearchVerifier):
    def __init__(self, model, case) -> None:
        super().__init__(model, case)
        self.verifier = Verifier(model, case, unstable_neurons_set=[], pair_wise_hinge=[], ho_hinge=[])
        
    def worker(self, veri_flag, ce, task_queue, output_queue, boundary_dict, boundary_list, pair_wise_hinge):
        try:
            while True:
                current_set = task_queue.get(timeout=0.001)
                previous_set = None
                if not veri_flag.value:
                    print('Verification failed!')
                    print('Segment counter example', ce)
                    break
                if task_queue.empty():
                    print("Worker received termination signal.")
                    break
                # print(f"Processing: {current_set}")
                if current_set in boundary_list:
                    continue
                res = solver_lp(self.model, current_set, SSpace=self.case.SSpace)
                if res.is_success():
                    output_queue.put(current_set)
                    hashable_d = tuple(sorted((k, tuple(v)) for k, v in current_set.items()))
                    
                    if previous_set is not None:
                        pair_wise_hinge.append([previous_set, current_set])
                    previous_set = current_set
                    
                    veri_flag_container, ce_item = self.verifier.seg_verification([current_set], reverse_flag = self.case.reverse_flag)
                    veri_flag.value = veri_flag_container
                    if ce_item is not None:
                        ce.append(ce_item)
                    if not veri_flag or len(ce) > 0:
                        print('Verification failed!')
                        print('Segment counter example', ce)
                        break
                    
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
            # print(f"Worker error: {e}")
            return
        finally:
            # print("Worker terminated.")
            return

    def BFS_parallel_with_verifier(self, root_node):
        manager = multiprocessing.Manager()
        veri_flag = manager.Value('b', True)
        ce = manager.list()
        task_queue = manager.Queue()
        output_queue = manager.Queue()
        boundary_dict = manager.list()
        boundary_list = manager.list()
        pair_wise_hinge = manager.list()

        num_workers = multiprocessing.cpu_count()
        # num_workers = 1
        # for _ in range(num_workers):
        #     task_queue.put(None)
            
        init_queue, pair_wise_hinge = self.BFS(root_node, termination=2*num_workers)
        for item in init_queue:
            task_queue.put(item)
            output_queue.put(item)
        
        workers = [multiprocessing.Process(target=self.worker, args=(veri_flag, ce, task_queue, output_queue, boundary_dict, boundary_list, pair_wise_hinge)) for _ in range(num_workers)]

        for w in workers:
            w.start()

        for w in workers:
            if veri_flag.value:
                w.join()
        # for w in workers:
        #     w.terminate()
        # all_finished = all(not w.is_alive() for w in workers)
        # if all_finished:
        #     print("All worker processes have finished.")
        print(veri_flag.value)
        if not veri_flag.value or len(ce) > 0:
            print('Verification failed!')
            print('Segment counter example output', ce)
            return False, ce, list(boundary_list), list(pair_wise_hinge)
        else:
            print(f"Boundary list size: {len(boundary_list)}, Pair-wise hinge size: {len(pair_wise_hinge)}")
            return veri_flag.value, ce, list(boundary_list), list(pair_wise_hinge)
    
    def BFS_with_Verifier(self, S):
        '''
        Breadth First Search
        Conduct breadth first search on the network to find the unstable neurons
        Initially, we start with the given set S, and then we find the neurons that are unstable at the boundary of S.
        To find the boundary of S, we conduct Filter_S_neighbour(S) to find the neurons that are unstable at the boundary of S.
        Then we add the neurons that are unstable at the boundary of S to the queue.
        We then iterate through the queue to find the neurons that are unstable at the boundary of the neurons that are unstable at the boundary of S.
        We continue this process until we find all the unstable neurons.
        '''
        queue = deque() 
        queue.append(S)
        # queue = S
        unstable_neurons = set()
        boundary_set = set()
        boundary_list = []
        previous_set = None
        pair_wise_hinge = []
        while queue:
            current_set = queue.popleft()
            # current_set = queue.pop(0)
            res = solver_lp(self.model, current_set, SSpace=self.case.SSpace) 
            # res = solver_lp(self.model, current_set)
            # print(res.is_success())
            if res.is_success():
                # add the current_set to the boundary_set (visited set)
                hashable_d = {k: tuple(v) for k, v in current_set.items()}
                tuple_representation = tuple(sorted(hashable_d.items()))
                if tuple_representation in boundary_set:
                    continue
                if previous_set is not None:
                    pair_wise_hinge.append([previous_set, current_set])
                boundary_set.add(tuple_representation)
                boundary_list.append(current_set)
                
                # verification
                veri_flag, ce = self.verifier.seg_verification([current_set], reverse_flag = self.case.reverse_flag)
                if not veri_flag:
                    print('Verification failed!')
                    print('Segment counter example', ce)
                    return False, ce, boundary_list, pair_wise_hinge
                
                # finding neighbours
                unstable_neighbours = self.Filter_S_neighbour(current_set)
                unstable_neighbours_S = self.Possible_S(current_set, unstable_neighbours)
                
                # check repeated set
                # for idx in range(len(unstable_neighbours_S)):
                for item in unstable_neighbours_S:    
                    hashable_u = {k: tuple(v) for k, v in item.items()}
                    tuple_representation = tuple(sorted(hashable_u.items()))
                    if tuple_representation in boundary_set:
                        # remove idx from unstable_neighbours_S
                        unstable_neighbours_S.remove(item)
                
                # add the unstable neighbours to the queue
                queue.extend(unstable_neighbours_S)
            else:
                continue
            previous_set = current_set
        return True, None, boundary_list, pair_wise_hinge

    
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
        if not veri_flag:
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
    Search_prog = SearchVerifierMT(model, case)
    veri_flag, ce = Search_prog.SV_CE(spt, uspt)
    if veri_flag:
        print('Verification successful!')
    else:
        print('Verification failed!')
        print('Counter example:', ce)
        