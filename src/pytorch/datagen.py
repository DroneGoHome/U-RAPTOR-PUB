import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

def get_binary_class_samples(main_path, class0_subpath, class1_subpath, num_samples):
    class0_dir = os.path.join(main_path, class0_subpath)
    class1_dir = os.path.join(main_path, class1_subpath)

    # Get all files in class0_dir (flat)
    class0_files = [
        os.path.join(class0_dir, f)
        for f in os.listdir(class0_dir)
        if os.path.isfile(os.path.join(class0_dir, f))
    ]
    
    if num_samples > len(class0_files):
        raise ValueError("Not enough samples in class0_dir")
    class0_samples = random.sample(class0_files, num_samples)

    # Get all files in all subfolders of class1_dir
    class1_files = []
    for root, dirs, files in os.walk(class1_dir):
        if root == class1_dir:
            continue  # skip the root itself
        for f in files:
            class1_files.append(os.path.join(root, f))
    print(f"Found {len(class1_files)} files in class1_dir subfolders")
    if num_samples > len(class1_files):
        raise ValueError("Not enough samples in class1_dir subfolders")
    class1_samples = random.sample(class1_files, num_samples)

    return class0_samples, class1_samples

def split_samples(class0_samples, class1_samples, val_ratio=0.2, seed=42):
    class0_train, class0_val = train_test_split(class0_samples, test_size=val_ratio, random_state=seed)
    class1_train, class1_val = train_test_split(class1_samples, test_size=val_ratio, random_state=seed)
    return (class0_train, class1_train), (class0_val, class1_val)

class BinaryImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def get_dataloader(class0_samples, class1_samples, batch_size=4, shuffle=True, transform=None):
    image_paths = class0_samples + class1_samples
    labels = [0] * len(class0_samples) + [1] * len(class1_samples)
    dataset = BinaryImageDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_dataloaders(main_path, class0_subpath, class1_subpath, num_samples, batch_size, val_ratio, train_transform=None, test_transform=None):
    """
    Utility to get train and test dataloaders for binary classification.
    
    Args:
        main_path (str): Path to the main directory containing class subdirectories.
        class0_subpath (str): Path to the class 0 subdirectory relative to main_path.
        class1_subpath (str): Path to the class 1 subdirectory relative to main_path.
        num_samples (int): Number of samples to select from each class.
        batch_size (int): Batch size for the dataloaders.
        val_ratio (float): Proportion of data to use for validation (0.0 to 1.0).
        train_transform (torchvision.transforms, optional): Transforms to apply to training data.
        test_transform (torchvision.transforms, optional): Transforms to apply to validation data.
    
    Returns:
        tuple: A tuple containing (train_loader, test_loader) as PyTorch DataLoader objects.
        
    Raises:
        ValueError: If num_samples exceeds available samples in either class.
    """
    class0_samples, class1_samples = get_binary_class_samples(main_path, class0_subpath, class1_subpath, num_samples)
    (class0_train, class1_train), (class0_val, class1_val) = split_samples(class0_samples, class1_samples, val_ratio)
    # Create DataLoaders
    train_loader = get_dataloader(class0_train, class1_train, batch_size=batch_size, transform=train_transform)
    test_loader = get_dataloader(class0_val, class1_val, batch_size=batch_size, transform=test_transform, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    main_path = '/mnt/d/Raptor/binary'
    class0_subpath = 'no_drone'
    class1_subpath = 'drones'

    class0_samples, class1_samples = get_binary_class_samples(main_path, class0_subpath, class1_subpath, 5)
    print("Class 0 Samples:")
    for sample in class0_samples:
        print(sample)
    print("\nClass 1 Samples:")
    for sample in class1_samples:
        print(sample)

    # Split into train and val
    (class0_train, class1_train), (class0_val, class1_val) = split_samples(class0_samples, class1_samples, val_ratio=0.2)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Train DataLoader
    train_loader = get_dataloader(class0_train, class1_train, batch_size=2, transform=transform)
    # Val DataLoader
    val_loader = get_dataloader(class0_val, class1_val, batch_size=2, transform=transform, shuffle=False)

    print("\nTrain batches:")
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}, labels: {labels}")

    print("\nValidation batches:")
    for images, labels in val_loader:
        print(f"Batch images shape: {images.shape}, labels: {labels}")