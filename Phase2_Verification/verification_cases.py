from Verifier.Verification import *
from SearchVerifier import SearchVerifier
def BC_Darboux():
    # BC Verification
    case = Darboux()
    architecture = [('linear', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase2_Verification/models/darboux_1_32.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    
    time_start = time.time()
    Search_prog = Search(model)
    Search_prog.Specify_point(torch.tensor([[[0.5, 1.5]]]), torch.tensor([[[-1, 0]]]))
    unstable_neurons_set, pairwise_hinge = Search_prog.BFS(Search_prog.S_init[0])
    ho_hinge = Search_prog.hinge_search(unstable_neurons_set, pairwise_hinge) 
    search_time = time.time() - time_start
    
    verifier = Verifier(model, case, unstable_neurons_set, pairwise_hinge, ho_hinge)
    veri_flag, ce = verifier.Verification(reverse_flag=True)
    verification_time = time.time() - time_start - search_time
    print('Search time:', search_time)
    print('Verification time:', verification_time)
    
def CBF_Obs(l, n):
    # CBF Verification
    case = ObsAvoid()
    hdlayers = []
    for layer in range(l):
        hdlayers.append(('relu', n))
    architecture = [('linear', 3)] + hdlayers + [('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load(f"Phase1_Scalability/models/obs_{l}_{n}.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    
    time_start = time.time()
    Search_prog = SearchVerifier(model, case)
    spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
    
    veri_flag, ce = Search_prog.SV_CE(spt, uspt)
    if veri_flag:
        print('Verification successful!')
    else:
        print('Verification failed!')
        print('Counter example:', ce)
    
    # Search_prog = Search(model)
    # Search_prog.Specify_point(spt, uspt)
    # unstable_neurons_set, pair_wise_hinge = Search_prog.BFS(Search_prog.S_init[0])
    # seg_search_time = time.time() - time_start
    # print('Seg Search time:', seg_search_time)
    # print('Num boundary seg is', len(unstable_neurons_set))
    
    # ho_hinge = Search_prog.hinge_search(unstable_neurons_set, pair_wise_hinge)
    # ho_hinge = Search_prog.hinge_search_3seg(unstable_neurons_set, pair_wise_hinge)
    # hinge_search_time = time.time() - time_start - seg_search_time
    # print('Hinge Search time:', hinge_search_time)
    # print('Num HO hinge is', len(ho_hinge))
    # if len(ho_hinge) > 0:
    #     print('Highest order is', np.max([len(ho_hinge[i]) for i in range(len(ho_hinge))]))
    # search_time = time.time() - time_start
    # print('Search time:', search_time)
    
    # o2_hinge = Search_prog.hinge_search_seg_comb(unstable_neurons_set, pair_wise_hinge, n=2)
    # hinge2_search_time = time.time() - time_start - seg_search_time
    # print('Hinge Search time:', hinge2_search_time)
    # print('Num HO hinge is', len(o2_hinge))
    # o2_hinge.extend(pair_wise_hinge)
    # o3_hinge = Search_prog.hinge_search_3seg(unstable_neurons_set, o2_hinge)
    
    # hinge3_search_time = time.time() - time_start - seg_search_time
    # print('Hinge Search time:', hinge3_search_time)
    # print('Num HO hinge is', len(o3_hinge))
    # if len(o3_hinge) > 0:
    #     print('Highest order is', np.max([len(o3_hinge[i]) for i in range(len(o3_hinge))]))
    # search_time = time.time() - time_start
    # print('Search time:', search_time)
    
    # verifier = Verifier(model, case, unstable_neurons_set, pair_wise_hinge, ho_hinge)
    # veri_flag, ce = verifier.Verification(reverse_flag=True, SMT_flag=True)
    # verification_time = time.time() - time_start - search_time
    # print('Search time:', search_time)
    # print('Verification time:', verification_time)

def CBF_LS(n):
    # CBF Verification
    case = LinearSat()
    architecture = [('linear', 6), ('relu', n), ('relu', n), ('linear', n), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load(f"./Phase2_Verification/models/satellitev1_2_{n}.pt")
    renamed_state_dict = model.wrapper_load_state_dict(trained_state_dict)
    # Load the renamed state dict into the model
    model.load_state_dict(renamed_state_dict, strict=True)
    model.merge_last_n_layers(2)
    
    time_start = time.time()
    Search_prog = Search(model)
    spt = torch.tensor([[[-1.2, -1.5, 1.1, 0.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    
    Search_prog.Specify_point(spt, uspt)
    unstable_neurons_set, pair_wise_hinge = Search_prog.BFS(Search_prog.S_init[0])
    seg_search_time = time.time() - time_start
    print('Seg Search time:', seg_search_time)
    print('Num boundary seg is', len(unstable_neurons_set))
    
    # ho_hinge = Search_prog.hinge_search(unstable_neurons_set, pair_wise_hinge)
    # ho_hinge = Search_prog.hinge_search_3seg(unstable_neurons_set, pair_wise_hinge)
    o2_hinge = Search_prog.hinge_search_seg_comb(unstable_neurons_set, pair_wise_hinge, n=2)
    hinge2_search_time = time.time() - time_start - seg_search_time
    print('Hinge Search time:', hinge2_search_time)
    print('Num HO hinge is', len(o2_hinge))
    o2_hinge.extend(pair_wise_hinge)
    o3_hinge = Search_prog.hinge_search_3seg(unstable_neurons_set, o2_hinge)
    
    hinge3_search_time = time.time() - time_start - seg_search_time
    print('Hinge Search time:', hinge3_search_time)
    print('Num HO hinge is', len(o3_hinge))
    if len(o3_hinge) > 0:
        print('Highest order is', np.max([len(o3_hinge[i]) for i in range(len(o3_hinge))]))
    search_time = time.time() - time_start
    print('Search time:', search_time)
    
    verifier = Verifier(model, case, unstable_neurons_set, pair_wise_hinge, ho_hinge)
    veri_flag, ce = verifier.Verification(reverse_flag=True)
    verification_time = time.time() - time_start - search_time
    
    print('Verification time:', verification_time)

def CBF_LS_SV(n):
    # # CBF Verification
    # n = 8
    # case = LinearSat()
    # architecture = [('linear', 6), ('relu', n), ('relu', n), ('linear', n), ('linear', 1)]
    # model = NNet(architecture)
    # trained_state_dict = torch.load(f"./Phase2_Verification/models/satellitev1_2_{n}.pt")
    # renamed_state_dict = model.wrapper_load_state_dict(trained_state_dict)
    # # Load the renamed state dict into the model
    # model.load_state_dict(renamed_state_dict, strict=True)
    # model.merge_last_n_layers(2)
    # spt = torch.tensor([[[-1.2, -1.5, 1.1, 0.0, 0.0, 0.0]]])
    # uspt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    
    # CBF Verification
    case = LinearSat()
    architecture = [('linear', 6), ('relu', n), ('relu', n), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load(f"Phase2_Verification/models/linear_satellite_hidden_32_epoch_50_reg_0.05.pt")
    # trained_state_dict = torch.load(f"Phase2_Verification/models/linear_satellite_layer_3_hidden_16_epoch_50_reg_0.pt")
    model.load_state_dict_from_sequential(trained_state_dict)
    spt = torch.tensor([[[2.0, 2.0, 2.0, 0.0, 0.0, 0.0]]]) * 5
    uspt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    
    # Search_prog = SearchVerifier(model, case)
    # spt = torch.tensor([[[2, 2, 1.1, 0.0, 0.0, 0.0]]])
    # uspt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    # spt = torch.tensor([[[2.2, 3.5, 4.1, 0.0, 0.0, 0.0]]])
    
    Search_prog = SearchVerifier(model, case)
    veri_flag, ce = Search_prog.SV_CE(spt, uspt)
    if veri_flag:
        print('Verification successful!')
    else:
        print('Verification failed!')
        print('Counter example:', ce)

if __name__ == "__main__":
    CBF_LS_SV(32)
    # CBF_LS_SV(8)
    # CBF_Obs(1, 128)
    