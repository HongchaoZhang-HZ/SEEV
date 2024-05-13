from Verifier.Verification import *
from collections import deque

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
            res = solver_lp(self.model, current_set) 
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

    def SV_CE(self, spt, uspt):
        # Search Verification and output Counter Example
        time_start = time.time()
        self.Specify_point(spt, uspt)
        unstable_neurons_set, pair_wise_hinge = self.BFS(self.S_init[0])
        # veri_flag, ce, unstable_neurons_set, pair_wise_hinge = self.BFS_with_Verifier(self.S_init[0])
        seg_search_time = time.time() - time_start
        print('Seg Search and Verification time:', seg_search_time)
        print('Num boundar seg is', len(unstable_neurons_set))
        # if not veri_flag:
        #     return False, ce
        
        # veri_flag, ce = self.verifier.hinge_verification(pair_wise_hinge, reverse_flag = self.case.reverse_flag)
        # if not veri_flag:
        #     return False, ce
        ho_hinge = self.hinge_search(unstable_neurons_set, pair_wise_hinge)
        veri_flag, ce = self.verifier.hinge_verification(ho_hinge, reverse_flag = self.case.reverse_flag)
        if not veri_flag:
            return False, ce
        hinge_search_time = time.time() - time_start - seg_search_time
        print('Hinge Search time:', hinge_search_time)
        print('Num HO hinge is', len(ho_hinge))
        if len(ho_hinge) > 0:
            print('Highest order is', np.max([len(ho_hinge[i]) for i in range(len(ho_hinge))]))
        search_time = time.time() - time_start
        print('Search time:', search_time)
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
        