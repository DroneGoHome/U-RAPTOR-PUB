import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from pathlib import Path

class NumpyArrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(224, 224), 
                 channels=3, file_extension='.pkl'):
        """
        Dataset for data arrays organized in class folders
        
        Args:
            root_dir (str): Root directory with class folders
            transform: Optional transforms to apply
            target_size (tuple): Target size for resizing (height, width)
            channels (int): Number of output channels (1 for grayscale, 3 for RGB)
            file_extension (str): File extension for data files (.npy or .pkl)
        
        Expected directory structure:
        root_dir/
        ├── class1/
        │   ├── file1.pkl
        │   ├── file2.pkl
        │   └── ...
        ├── class2/
        │   ├── file1.pkl
        │   └── ...
        └── ...
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.channels = channels
        self.file_extension = file_extension
        
        # Get class names (folder names)
        self.classes = sorted([d.name for d in self.root_dir.iterdir() 
                              if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Build file list
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for file_path in class_dir.glob(f'*{self.file_extension}'):
                self.samples.append((file_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Load data based on file extension
        if str(file_path).endswith('.pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:  # Default to numpy for backward compatibility
            data = np.load(file_path)
        
        # Convert to tensor and add channel dimension if needed
        if data.ndim == 2:
            data = torch.from_numpy(data).float().unsqueeze(0)  # Add channel dim
        else:
            data = torch.from_numpy(data).float()
        
        # Resize to target size
        if data.shape[-2:] != self.target_size:
            data = F.interpolate(data.unsqueeze(0), size=self.target_size, 
                               mode='bilinear', align_corners=False).squeeze(0)
        
        # Handle channel dimension
        if self.channels == 3 and data.shape[0] == 1:
            # Convert grayscale to RGB by repeating channels
            data = data.repeat(3, 1, 1)
        elif self.channels == 1 and data.shape[0] == 3:
            # Convert RGB to grayscale
            data = data.mean(dim=0, keepdim=True)
        
        # Apply transforms if provided
        if self.transform:
            data = self.transform(data)
        
        return data, label

def create_data_loaders(root_dir, batch_size=16, test_ratio=0.2, transform=None, 
                        target_size=(224, 224), channels=3, num_workers=4, 
                        shuffle_train=True, seed=42, file_extension='.pkl'):
    """
    Create train and test data loaders with specified split ratio
    
    Args:
        root_dir (str): Root directory with class folders
        batch_size (int): Batch size for data loaders
        test_ratio (float): Proportion of data to use for testing (0.0 to 1.0)
        transform: Optional transforms to apply
        target_size (tuple): Target size for resizing
        channels (int): Number of output channels
        num_workers (int): Number of worker processes for data loading
        shuffle_train (bool): Whether to shuffle the training data
        seed (int): Random seed for reproducibility
        file_extension (str): File extension for data files (.npy or .pkl)
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create the dataset
    full_dataset = NumpyArrayDataset(
        root_dir=root_dir,
        transform=transform,
        target_size=target_size,
        channels=channels,
        file_extension=file_extension
    )
    
    # Ensure dataset is not empty
    if len(full_dataset) == 0:
        raise ValueError(f"No valid files found in {root_dir}. Make sure the directory exists and contains class folders with {file_extension} files.")
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_ratio)
    train_size = dataset_size - test_size
    
    # Split the dataset
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset split: {train_size} training samples, {test_size} test samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NumpyArrayDataset functionality")
    parser.add_argument('--data_path', type=str, default='/mnt/d/Raptor/binary/drones', help='Main data directory')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--file_ext', type=str, default='.pkl', help="File extension for data files (e.g., .pkl, .npy)")
    args = parser.parse_args()
    
    # Verify the data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist!")
        exit(1)
    
    print(f"Searching for {args.file_ext} files in {args.data_path}...")
    
    # Create data loaders with 80/20 split (default test_ratio=0.2)
    try:
        train_loader, test_loader = create_data_loaders(
            root_dir=args.data_path,
            batch_size=args.batch_size,
            file_extension=args.file_ext
        )
        
        # Print dataset information
        dataset = train_loader.dataset.dataset  # Get the original dataset through the random_split wrapper
        print(f"Classes: {dataset.classes}")
        print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        # Test loading a batch
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
            break
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Print directory structure for debugging
        print("\nDirectory structure:")
        for root, dirs, files in os.walk(args.data_path):
            level = root.replace(args.data_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            if files:
                for f in files[:5]:  # Show first 5 files only
                    print(f"{indent}    {f}")
                if len(files) > 5:
                    print(f"{indent}    ... ({len(files) - 5} more files)")