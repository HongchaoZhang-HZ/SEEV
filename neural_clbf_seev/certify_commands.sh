
#Darboux
python certify_cbf.py --system_name darboux --cbf_hidden_layers 2 --cbf_hidden_size 256 --model_path models/darboux_2_256.pt
python certify_cbf.py --system_name darboux --cbf_hidden_layers 2 --cbf_hidden_size 512 --model_path models/darboux_2_512.pt

# Obs Avoid
python certify_cbf.py --system_name obs_avoid --cbf_hidden_layers 2 --cbf_hidden_size 16 --model_path models/obs_avoid_commit_18b5c40_layers_2_size_16_seed_111_certify_True_reg_0.0.pt
python certify_cbf.py --system_name obs_avoid --cbf_hidden_layers 4 --cbf_hidden_size 8 --model_path models/obs_avoid_commit_18b5c40_layers_4_size_8_seed_111_certify_True_reg_0.0.pt
python certify_cbf.py --system_name obs_avoid --cbf_hidden_layers 4 --cbf_hidden_size 16 --model_path models/obs_avoid_commit_18b5c40_layers_4_size_16_seed_111_certify_True_reg_0.0.pt
python certify_cbf.py --system_name obs_avoid --cbf_hidden_layers 1 --cbf_hidden_size 128 --model_path models/obs_1_128.pt


# Linear satellite
python certify_cbf.py --system_name linear_satellite --cbf_hidden_layers 2 --cbf_hidden_size 8 --model_path models/linear_satellite_commit_c8861a8_layers_2_size_8_seed_111_certify_True_reg_0.0.pt
python certify_cbf.py --system_name linear_satellite --cbf_hidden_layers 4 --cbf_hidden_size 8 --model_path models/linear_satellite_commit_c8861a8_layers_4_size_8_seed_111_certify_True_reg_0.0.pt

# HighOrd8
python certify_cbf.py --system_name high_o --cbf_hidden_layers 2 --cbf_hidden_size 8 --model_path models/high_o_commit_2fd0d11_layers_2_size_8_seed_222_certify_True.pt
python certify_cbf.py --system_name high_o --cbf_hidden_layers 2 --cbf_hidden_size 16 --model_path models/high_o_commit_2fd0d11_layers_2_size_16_seed_111_certify_True.pt
python certify_cbf.py --system_name high_o --cbf_hidden_layers 4 --cbf_hidden_size 16 --model_path models/high_o_commit_2fd0d11_layers_4_size_16_seed_222_certify_True.pt