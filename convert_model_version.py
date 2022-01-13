import torch

model_path = "./deploy_model/unified_large_5000_0.538/5000/mp_rank_00_model_states.pt"
output_path = "./deploy_model/unified_large_5000_0.538/5000/mp_rank_00_model_states-1.4.pt"

if __name__ == "__main__":
    print("Loading ...")
    model = torch.load(model_path, map_location='cpu')
    print(model.keys())
    print("Writing ...")
    torch.save(model, output_path, _use_new_zipfile_serialization=False)