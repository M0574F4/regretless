# dataloaders.py
import os
import yaml
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
from modem import QPSKModulator, QPSKDemodulator
import math

#######################################################
# Set seeds
#######################################################
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#######################################################
# Dataset
#######################################################
class RFDataset(Dataset):
    """
    Returns a mixture of interference + QPSK signal, plus
    the QPSK bits, chosen SINR, and index for each item.
    """
    def __init__(
        self,
        data_dict,
        dataset_frame_pairs,
        win_len,
        qpsk_modulator,
        sinr_db_min=0,
        sinr_db_max=20,
        bits_per_frame=4096
    ):
        """
        Args:
            data_dict            : dictionary of signals loaded from .pt,
                                   each entry has shape [num_frames, num_samples, 2]
                                   (where the last dim is [real, imag]).
            dataset_frame_pairs  : list of (dataset_name, frame_idx)
            win_len              : length (in samples) for the interference window
            qpsk_modulator       : an instance of QPSKModulator that outputs
                                   shape [1, num_samples, 2] (real, imag)
            sinr_db_min          : lower bound on random SINR
            sinr_db_max          : upper bound on random SINR
            bits_per_frame       : how many bits to generate for each QPSK
                                   typically ensure sps * (#symbols) >= win_len
        """
        super().__init__()
        self.data_dict = data_dict
        self.dataset_frame_pairs = dataset_frame_pairs
        self.win_len = win_len

        # QPSK modulator
        self.qpsk_mod = qpsk_modulator
        
        self.fixed_sinr_db = None           # <--- This is our "global" override

        self.sinr_db_min = sinr_db_min
        self.sinr_db_max = sinr_db_max

        self.bits_per_frame = bits_per_frame
        self.sps = qpsk_modulator.sps

    def __len__(self):
        return len(self.dataset_frame_pairs)

    def set_fixed_sinr_db(self, sinr_db):
        """Set a global sinr_db used for every sample."""
        self.fixed_sinr_db = sinr_db

    def __getitem__(self, idx, sinr_db=None):
        dataset_name, frame_idx = self.dataset_frame_pairs[idx]

        # 1) Get random interference window
        frame = self.data_dict[dataset_name][frame_idx]
        num_samples = frame.shape[0]
        if num_samples < self.win_len:
            raise ValueError(
                f"Frame size {num_samples} is smaller than win_len {self.win_len}"
            )

        start_idx = random.randint(0, num_samples - self.win_len)
        interference_window = frame[start_idx : start_idx + self.win_len]  # [win_len, 2]

        # --- Multiply by random phase -----------------------------------
        phase = 2.0 * math.pi * random.random()
        cos_phase = math.cos(phase)
        sin_phase = math.sin(phase)

        # Rotate the real & imag parts
        real = interference_window[..., 0]  # shape: [win_len]
        imag = interference_window[..., 1]  # shape: [win_len]

        real_rot = real * cos_phase - imag * sin_phase
        imag_rot = real * sin_phase + imag * cos_phase

        # Make interference channel-first: [2, win_len]
        interference_window_rotated = torch.stack((real_rot, imag_rot), dim=0)

        # 2) Generate QPSK signal (initially [win_len, 2])
        bits_in = torch.randint(0, 2, (1, self.bits_per_frame))
        qpsk_waveform = self.qpsk_mod(bits_in).squeeze(0)  # => shape [win_len, 2]

        # Make QPSK channel-first: [2, win_len]
        qpsk_waveform = qpsk_waveform.permute(1, 0)

        # 3) Compute powers
        # Sum over channel dimension, then average over time dimension
        power_interf = torch.mean(torch.sum(interference_window_rotated ** 2, dim=0)).item()
        power_qpsk = torch.mean(torch.sum(qpsk_waveform ** 2, dim=0)).item()

        # Avoid zero power
        if power_interf < 1e-12:
            power_interf = 1e-12
        if power_qpsk < 1e-12:
            power_qpsk = 1e-12

        # 4) Compute random (or fixed) SINR scaling
        if self.fixed_sinr_db is not None:
            sinr_db = self.fixed_sinr_db
            self.fixed_sinr_db = None
            print(f'fixed sinr used!')
        else:
            sinr_db = random.uniform(self.sinr_db_min, self.sinr_db_max)

        sinr_lin = 10.0 ** (sinr_db / 10.0)
        scaling = math.sqrt(power_qpsk / (power_interf * sinr_lin))

        # 5) Scale the interference
        interference_scaled = interference_window_rotated * scaling

        # 6) Build the mixture: [2, win_len]
        mixture = interference_scaled + qpsk_waveform

        return {
            'input': mixture,            # [2, win_len]
            'label': qpsk_waveform,      # [2, win_len]
            'bits': bits_in,
            'sinr_db': sinr_db,
            'metadata': (dataset_name, frame_idx)
        }


#######################################################
# Helper functions
#######################################################
def make_dataset_frame_pairs(fold_split_dict, split_name):
    dataset_frame_pairs = []
    for dataset_name, split_info in fold_split_dict.items():
        indices = split_info.get(split_name, [])
        for idx in indices:
            dataset_frame_pairs.append((dataset_name, idx))
    return dataset_frame_pairs

def split_train_into_train_val(fold_split, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    for dataset_name, indices_dict in fold_split.items():
        if "train_indices" in indices_dict:
            train_inds = indices_dict["train_indices"]
            rng.shuffle(train_inds)
            val_size = int(len(train_inds) * val_ratio)
            val_inds = train_inds[:val_size]
            new_train_inds = train_inds[val_size:]
            indices_dict["train_indices"] = new_train_inds
            indices_dict["val_indices"] = val_inds

#######################################################
# Main get_dataloaders
#######################################################
def get_dataloaders(config_all):
    """
    Build and return train_loader, val_loader, test_loader in a reproducible way.

    config example:
      {
        "random_seed": 2023,
        "folds_yaml_path": "dataset/folds.yaml",
        "fold_idx": 0,
        "data_pt_path": "dataset/combined_signals.pt",
        "win_len": 1024,
        "batch_size": 32,
        "num_workers": 4,
        "use_balanced_sampler": True,
        "val_ratio": 0.2,
        "split_names": ["train_indices","test_indices"],
        # QPSK-related
        "sps": 32,
        "span": 8,
        "beta": 0.5,
        "bits_per_frame": 4096,
        "sinr_db_min": 0,
        "sinr_db_max": 20
      }
    """
    config = config_all["dataset"]
    # Set seeds
    seed_everything(config_all["random_seed"])

    # Load data dict (each entry is [num_frames, num_samples, 2])
    data_dict = torch.load(config["data_pt_path"])

    # Load fold info (train/test) from YAML
    with open(config["folds_yaml_path"], "r") as f:
        folds_dict = yaml.safe_load(f)

    fold_idx = config["fold_idx"]
    fold_split = folds_dict[fold_idx]

    # If there's no val split, create one
    found_val_split = any("val_indices" in v for v in fold_split.values())
    if not found_val_split:
        val_ratio = config["val_ratio"]
        split_train_into_train_val(fold_split, val_ratio=val_ratio,
                                   seed=config_all["random_seed"])

    # Create QPSK modulator
    qpsk_mod = QPSKModulator(
        sps=config["sps"], 
        span=config["span"], 
        beta=config["beta"]
    )

    # -------------------------------------------------------
    # Create RFDataset objects
    # -------------------------------------------------------
    win_len = config["win_len"]
    desired_splits = ["train_indices","val_indices","test_indices"]
    dataset_objects = {}
    for split_name in desired_splits:
        # skip if that split doesn't exist in fold_split or not in config["split_names"]
        if any(split_name in x for x in fold_split.values()) or (split_name in config["split_names"]):
            dataset_frame_pairs = make_dataset_frame_pairs(fold_split, split_name)

            # Here we override bits_per_frame so that
            # QPSK wave can handle at least 'win_len' samples.
            # Typically 2 * win_len / sps = #bits needed for QPSK (bits per symbol = 2).
            ds = RFDataset(
                data_dict=data_dict,
                dataset_frame_pairs=dataset_frame_pairs,
                win_len=win_len,
                qpsk_modulator=qpsk_mod,
                sinr_db_min=config["sinr_db_min"],
                sinr_db_max=config["sinr_db_max"],
                bits_per_frame=2 * win_len // config["sps"]
            )
            dataset_objects[split_name] = ds

    # -------------------------------------------------------
    # Weighted sampler for train
    # -------------------------------------------------------
    use_balanced_sampler = config["use_balanced_sampler"]
    train_sampler = None
    if "train_indices" in dataset_objects and use_balanced_sampler:
        train_ds = dataset_objects["train_indices"]
        dataset_name_to_label = {
            "CommSignal2": 0,
            "CommSignal3": 1,
            "CommSignal5G1": 2,
            "EMISignal1": 3
        }
        train_labels = []
        for (dataset_name, _) in train_ds.dataset_frame_pairs:
            train_labels.append(dataset_name_to_label[dataset_name])

        label_counts = Counter(train_labels)
        weights_for_label = {}
        for lbl, cnt in label_counts.items():
            weights_for_label[lbl] = 1.0 / float(cnt)
        sample_weights = [weights_for_label[lbl] for lbl in train_labels]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    # -------------------------------------------------------
    # Build DataLoaders
    # -------------------------------------------------------
    g = torch.Generator()
    g.manual_seed(config_all["random_seed"])

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    dataloaders = {}
    for split_name, ds_obj in dataset_objects.items():
        if split_name == "train_indices" and train_sampler is not None:
            dl = DataLoader(ds_obj,
                            sampler=train_sampler,
                            drop_last=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            worker_init_fn=seed_worker,
                            generator=g,
                            pin_memory=True)
        else:
            dl = DataLoader(ds_obj,
                            shuffle=False,
                            drop_last=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            worker_init_fn=seed_worker,
                            generator=g,
                            pin_memory=True)
        dataloaders[split_name] = dl

    # Return in a convenient order
    train_loader = dataloaders["train_indices"]
    val_loader   = dataloaders["val_indices"]
    test_loader  = dataloaders["test_indices"]

    return train_loader, val_loader, test_loader
