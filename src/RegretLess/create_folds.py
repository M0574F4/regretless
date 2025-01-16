# create_folds.py
import torch
import random
import yaml
import numpy as np

def create_folds(data_dict, n_folds=5, seed=42):
    """
    data_dict: {
      "CommSignal2": tensor_or_array_of_shape (100, ...),
      "CommSignal3": tensor_or_array_of_shape (139, ...),
      ...
    }

    Returns a dict of folds:
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

    for name, frames in data_dict.items():
        num_frames = frames.shape[0]
        indices = list(range(num_frames))
        random.shuffle(indices)

        # Split indices into n_folds distinct subsets
        fold_size = num_frames // n_folds  # integer division
        subsets = []
        for i in range(n_folds):
            start_idx = i * fold_size
            if i < n_folds - 1:
                end_idx = (i + 1) * fold_size
            else:
                # Last fold takes the remainder
                end_idx = num_frames
            fold_subset = indices[start_idx:end_idx]
            subsets.append(fold_subset)

        # Now subsets is a list of length n_folds, each a chunk of indices
        # Build folds so that fold i uses subsets[i] as test, the rest as train
        for fold_idx in range(n_folds):
            if fold_idx not in folds:
                folds[fold_idx] = {}
            if name not in folds[fold_idx]:
                folds[fold_idx][name] = {}

            test_indices = subsets[fold_idx]
            train_indices = []
            for j in range(n_folds):
                if j != fold_idx:
                    train_indices.extend(subsets[j])

            folds[fold_idx][name]["test_indices"] = test_indices
            folds[fold_idx][name]["train_indices"] = train_indices

    return folds

def main():
    # Load the .pt file which has the combined data_dict
    data_dict = torch.load("dataset/combined_signals.pt")
    
    # Create 5 folds
    folds = create_folds(data_dict, n_folds=5, seed=2023)

    # Save to YAML
    with open("dataset/folds.yaml", "w") as f:
        yaml.dump(folds, f)

    print("Saved folds.yaml")

if __name__ == "__main__":
    main()
