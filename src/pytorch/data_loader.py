# %%
import os
import sys
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Import transforms


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

from data.dataset import load_json, load_pickle

class SpectogramDataset(Dataset):
    def __init__(self, meta_data_path=None, transform=None):
        """
        Args:
            data: A list or array of your input data.
            targets: A list or array of your target labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = os.path.dirname(meta_data_path)
        self.meta_data = load_json(meta_data_path)
        self.sample_paths = [os.path.join(self.dataset_path, sample['signal_path']) for sample in self.meta_data]
        all_string_targets = [sample['model'] for sample in self.meta_data]
        self.unique_labels = sorted(list(set(all_string_targets)))
        self.label_to_int = {label: i for i, label in enumerate(self.unique_labels)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        self.targets = [self.label_to_int[label] for label in all_string_targets] # Store targets as integers

        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.sample_paths)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = load_pickle(self.sample_paths[idx]).astype(np.float32)
        sample = torch.from_numpy(sample).float()
        # min_val = sample.min()
        # max_val = sample.max()
        # if max_val - min_val > 1e-6: # Avoid division by zero
        #     sample = ((sample - min_val) / (max_val - min_val) * 255)

        # Load the spectrogram or any other data processing here

        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target

# %%
if __name__ == '__main__':
    meta_data_path = '../../data/Related work from Rowan/dataset/meta_data.json'
    print(f"Attempting to load metadata from: {meta_data_path}"),
    custom_dataset = SpectogramDataset(meta_data_path=meta_data_path)

    print("Dataset initialized.")
    print(f"Number of samples in dataset: {len(custom_dataset)}")

    # Test __getitem__ for the first sample",
    if len(custom_dataset) > 0:
        print("Fetching first sample directly from dataset:")
        sample, target = custom_dataset[0]
        print("Sample type: {type(sample)}")
        if isinstance(sample, torch.Tensor):
            print(f"Sample shape: {sample.shape}")
            print(f"Sample dtype: {sample.dtype}")
        else:
            print(f"Sample data: {sample}") # Print sample if not a tensor to see its structure
        print(f"Target type: {type(target)}")
        print(f"Target: {target}")
    else:
        print("Dataset is empty, cannot fetch first sample.")
    # Test DataLoader",
    if len(custom_dataset) > 0:
        print("Testing DataLoader...")
        # You might need to adjust batch_size depending on your data and memory",
        # If your targets are strings, you'll need a custom collate_fn for the DataLoader",
        # or convert targets to numerical representations (e.g., class indices) in __getitem__.",
        # For now, assuming targets can be batched by default DataLoader or are numerical.",
        
        data_loader = DataLoader(custom_dataset, batch_size=4, shuffle=True)
        
        # Get one batch from the DataLoader",
        samples_batch, targets_batch = next(iter(data_loader))
        
        print("Fetched one batch from DataLoader:")
        print(f"Samples batch type: {type(samples_batch)}")
        if isinstance(samples_batch, torch.Tensor):
            print(f"Samples batch shape: {samples_batch.shape}") # Expected: (batch_size, C, H, W)
            print(f"Samples batch dtype: {samples_batch.dtype}")
        else:
            print(f"Samples batch data: {samples_batch}")
        print(f"Targets batch type: {type(targets_batch)}")
        # If targets are strings, this will be a list/tuple of strings
        # If targets are numerical, this will be a tensor
        print(f"Targets batch: {targets_batch}")
        if isinstance(targets_batch, torch.Tensor):
            print(f"Targets batch shape: {targets_batch.shape}")


# %%
