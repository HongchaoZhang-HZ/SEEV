from Verifier.Verification import *
from collections import deque
from Modules.Function import *
class SearchVerifier(Search):
    def __init__(self, model, case) -> None:
        super().__init__(model, case)
        self.verifier = Verifier(model, case, unstable_neurons_set=[], pair_wise_hinge=[], ho_hinge=[])
        
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
        veri_flag, ce, unstable_neurons_set, pair_wise_hinge = self.BFS_with_Verifier(self.S_init[0])
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
        
        if len(prob_S_checklist) >= len(pair_wise_hinge)/2:
            veri_flag, ce = self.verifier.hinge_verification(o2_hinge, reverse_flag=self.case.reverse_flag)
            if not veri_flag:
                return False, ce
        o2_hinge = self.hinge_search_seg_comb(prob_S_checklist, pair_wise_hinge, n=2)
        hinge2_search_time = time.time() - time_start - seg_search_time
        print('O2 Hinge Search time:', hinge2_search_time)
        veri_flag, ce = self.verifier.hinge_verification(o2_hinge, reverse_flag=self.case.reverse_flag)
        hinge2_verification_time = time.time() - time_start - hinge2_search_time
        print('Pair-wise Hinge Verification time:', hinge2_verification_time)
        if not veri_flag:
            return False, ce
        
        o3_hinge = self.hinge_search_3seg(prob_S_checklist, o2_hinge)
        hinge3_search_time = time.time() - time_start - hinge2_verification_time
        print('O3 Hinge Search time:', hinge3_search_time)
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
    trained_state_dict = torch.load(f"Phase1_Scalability/models/obs_{l}_{n}.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    
    spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
    # Search Verification and output Counter Example
    Search_prog = SearchVerifier(model, case)
    veri_flag, ce = Search_prog.SV_CE(spt, uspt)
    if veri_flag:
        print('Verification successful!')
    else:
        print('Verification failed!')
        print('Counter example:', ce)
        