import torch
from torch.utils.data import Dataset
import numpy as np

class SignalDataset(Dataset):
    def __init__(self, data_dict, dataset_frame_pairs, win_len):
        """
        data_dict:  {"CommSignal2": np.ndarray, "CommSignal3": np.ndarray, ...}
        dataset_frame_pairs: list of tuples (dataset_name, frame_idx),
                             e.g. [("CommSignal2", 0), ("CommSignal2", 1), ...]
        win_len: int, number of samples in the random window
        """
        self.data_dict = data_dict
        self.dataset_frame_pairs = dataset_frame_pairs
        self.win_len = win_len

    def __len__(self):
        return len(self.dataset_frame_pairs)

    def __getitem__(self, idx):
        dataset_name, frame_idx = self.dataset_frame_pairs[idx]
        
        # entire frame: shape = (num_samples,) or (num_samples, ) complex
        frame = self.data_dict[dataset_name][frame_idx]  
        
        num_samples = frame.shape[0]
        if num_samples < self.win_len:
            raise ValueError(f"Frame size {num_samples} is smaller than win_len {self.win_len}")

        start = np.random.randint(0, num_samples - self.win_len + 1)
        window = frame[start:start + self.win_len]

        # If complex, you might want to split real/imag or keep as complex
        # E.g., convert to 2D [real, imag]:
        window = np.stack([window.real, window.imag], axis=-1)  # shape = (win_len, 2)

        # Return as torch.Tensor
        return torch.from_numpy(window).float()
