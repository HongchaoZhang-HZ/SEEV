from Verifier.Verification import *
from SearchVerifier import SearchVerifier
from SearchVerifierMT import SearchVerifierMT

def CBF_Obs(l, n):
    # CBF Verification
    case = ObsAvoid()
    hdlayers = []
    for layer in range(l):
        hdlayers.append(('relu', n))
    architecture = [('linear', 3)] + hdlayers + [('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load(f"models/obs_{l}_{n}.pt")
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

def CBF_LS_SV(n):
    # CBF Verification
    case = LinearSat()
    architecture = [('linear', 6), ('relu', n), ('relu', n), ('relu', n), ('relu', n), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load(f"models/linear_satellite_layer_4_hidden_8_epoch_50_reg_0.1.pt")
    # trained_state_dict = torch.load(f"models/linear_satellite_layer_4_hidden_8_epoch_50_reg_0.1.pt")
    model.load_state_dict_from_sequential(trained_state_dict)
    spt = torch.tensor([[[2.0, 2.0, 2.0, 0.0, 0.0, 0.0]]]) * 5
    uspt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    
    # Search_prog = SearchVerifier(model, case)
    # spt = torch.tensor([[[2, 2, 1.1, 0.0, 0.0, 0.0]]])
    # uspt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    # spt = torch.tensor([[[2.2, 3.5, 4.1, 0.0, 0.0, 0.0]]])
    
    Search_prog = SearchVerifierMT(model, case)
    veri_flag, ce = Search_prog.SV_CE(spt, uspt)
    if veri_flag:
        print('Verification successful!')
    else:
        print('Verification failed!')
        print('Counter example:', ce)
    
if __name__ == "__main__":
    # CBF_LS_SV(8)
    CBF_Obs(1, 128)
