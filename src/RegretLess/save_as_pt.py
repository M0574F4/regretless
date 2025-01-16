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
        frames = f["dataset"][:]  # gets a numpy array of complex numbers (if complex)
    
    # Convert the complex numpy array into a torch tensor with two channels.
    # Get the real and imaginary parts as separate arrays:
    real_part = frames.real  # shape: [num_frames, num_samples]
    imag_part = frames.imag  # shape: [num_frames, num_samples]

    # Convert to torch tensors; you can specify a desired dtype such as torch.float32.
    real_tensor = torch.tensor(real_part, dtype=torch.float32)
    imag_tensor = torch.tensor(imag_part, dtype=torch.float32)
    
    # Stack them along a new dimension (e.g. the last dimension).
    # This produces a tensor with shape: [num_frames, num_samples, 2]
    tensor_frames = torch.stack([real_tensor, imag_tensor], dim=-1)

    data_dict[name] = tensor_frames
    
    # Optional: Verification printout
    print(f"{name}:")
    print(f"  Type: {type(tensor_frames)}")  
    print(f"  Dtype: {tensor_frames.dtype}")
    print(f"  Shape: {tensor_frames.shape}")
    print("-" * 40)

torch.save(data_dict, "dataset/combined_signals.pt")
print("Saved combined_signals.pt")
