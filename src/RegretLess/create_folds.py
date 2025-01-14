import torch
import random
import yaml  # or json
import numpy as np

def create_folds(data_dict, n_folds=5, seed=42):
    """
    data_dict: {"CommSignal2": array_of_shape (100, 43560),
                "CommSignal3": array_of_shape (139, 260000),
                ...
               }
    Returns a dictionary like:
    {
        0: {
            "CommSignal2": {
                "train_indices": [...],
                "test_indices": [...]
            },
            "CommSignal3": {
                "train_indices": [...],
                "test_indices": [...]
            },
            ...
        },
        1: { ... },
        ...
    }
    """
    random.seed(seed)
    folds = {}

    for fold_idx in range(n_folds):
        fold_split = {}
        for name, frames in data_dict.items():
            num_frames = frames.shape[0]
            indices = list(range(num_frames))
            random.shuffle(indices)

            # 1/5 for test
            split_size = num_frames // 5  # integer division
            test_indices = indices[:split_size]
            train_indices = indices[split_size:]

            fold_split[name] = {
                "train_indices": train_indices,
                "test_indices": test_indices
            }
        folds[fold_idx] = fold_split

    return folds

def main():
    # Load the .pt file which has the combined data_dict
    data_dict = torch.load("dataset/combined_signals.pt")
    
    # Create 5 folds
    folds = create_folds(data_dict, n_folds=5, seed=2023)

    # Save to YAML (or JSON)
    with open("dataset/folds.yaml", "w") as f:
        yaml.dump(folds, f)
    print("Saved folds.yaml")

if __name__ == "__main__":
    main()
