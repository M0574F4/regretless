import h5py
import torch
import numpy as np

file_paths = {
    "CommSignal2": "/project_ghent/Mostafa/icassp2024rfchallenge/icassp2024rfchallenge-0.2.0/dataset/interferenceset_frame/CommSignal2_raw_data.h5",
    "CommSignal3": "/project_ghent/Mostafa/icassp2024rfchallenge/icassp2024rfchallenge-0.2.0/dataset/interferenceset_frame/CommSignal3_raw_data.h5",
    "CommSignal5G1": "/project_ghent/Mostafa/icassp2024rfchallenge/icassp2024rfchallenge-0.2.0/dataset/interferenceset_frame/CommSignal5G1_raw_data.h5",
    "EMISignal1": "/project_ghent/Mostafa/icassp2024rfchallenge/icassp2024rfchallenge-0.2.0/dataset/interferenceset_frame/EMISignal1_raw_data.h5"
}

data_dict = {}

for name, path in file_paths.items():
    with h5py.File(path, 'r') as f:
        frames = f["dataset"][:]  # shape = (num_frames, num_samples)
    data_dict[name] = frames  # store in a dict in memory

# Save to a .pt file
torch.save(data_dict, "combined_signals.pt")
print("Saved combined_signals.pt")
