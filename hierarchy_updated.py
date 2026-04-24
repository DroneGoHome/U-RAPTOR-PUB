import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from Moe.src.constants import hierarchy
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from Moe.generator_data_loader import SpectogramDataset
from tqdm import tqdm

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

class HierarchyUtils:
    """Utility class for handling hierarchical classification mappings"""
    
    def __init__(self, hierarchy_dict):
        self.hierarchy = hierarchy_dict
        self.level_mappings = self._create_level_mappings()
        self.class_to_path = self._create_class_to_path()
        
    def _create_level_mappings(self):
        """Create mappings for each level of the hierarchy"""
        mappings = {}
        
        # Level 0: Make (top level)
        make_labels = list(self.hierarchy.keys())
        mappings['modulation'] = {make: idx for idx, make in enumerate(make_labels)}
        
        # Level 1: Type (middle level)
        type_labels = []
        for make in self.hierarchy:
            for type_name in self.hierarchy[make]:
                if type_name not in type_labels:
                    type_labels.append(type_name)
        mappings['protocol'] = {type_name: idx for idx, type_name in enumerate(type_labels)}
        
        # Level 2: Class (leaf level)
        class_labels = []
        for make in self.hierarchy:
            for type_name in self.hierarchy[make]:
                for class_name in self.hierarchy[make][type_name]:
                    class_labels.append(class_name)
        mappings['label'] = {class_name: idx for idx, class_name in enumerate(class_labels)}
        
        return mappings
    
    def _create_class_to_path(self):
        """Create mapping from class name to full hierarchical path"""
        class_to_path = {}
        for make in self.hierarchy:
            for type_name in self.hierarchy[make]:
                for class_name in self.hierarchy[make][type_name]:
                    class_to_path[class_name] = {
                        'modulation': make,
                        'protocol': type_name,
                        'label': class_name,
                        'modulation_label': self.level_mappings['modulation'][make],
                        'protocol_label': self.level_mappings['protocol'][type_name],
                        'model_label': self.level_mappings['label'][class_name]
                    }
        return class_to_path
    
    def get_hierarchical_labels(self, class_name: str) -> Tuple[int, int, int]:
        """Get hierarchical labels for a given class name"""
        if class_name not in self.class_to_path:
            raise ValueError(f"Class {class_name} not found in hierarchy")
        
        path = self.class_to_path[class_name]
        return path['make_label'], path['type_label'], path['class_label']
    
    def get_num_classes_per_level(self) -> Tuple[int, int, int]:
        """Get number of classes at each level"""
        return (
            len(self.level_mappings['modulation']),
            len(self.level_mappings['protocol']),
            len(self.level_mappings['label'])
        )

class HierarchicalSpectrogramDataset(Dataset):
    """Wrapper for SpectrogramDataset to add hierarchical labels"""
    
    def __init__(self, spectogram_dataset: SpectogramDataset, hierarchy_utils: HierarchyUtils):
        self.spectogram_dataset = spectogram_dataset
        self.hierarchy_utils = hierarchy_utils
        
    def __len__(self):
        return len(self.spectogram_dataset)
    
    def __getitem__(self, idx):
        # Get the original data from SpectrogramDataset
        image, original_label = self.spectogram_dataset[idx]

        # Access the metadata directly from the SpectrogramDataset
        meta_entry = self.spectogram_dataset.master_metadata[idx]
        
        # Extract hierarchical labels directly from metadata
        # Since cleaned_meta_data.json already has these labels
        make_label = meta_entry.get('modulation_label', -1)
        type_label = meta_entry.get('protocol_label', -1)
        class_label = meta_entry.get('model_label', -1)

        # Validate that we have valid labels
        if make_label == -1 or type_label == -1 or class_label == -1:
            print(f"Warning: Missing hierarchical labels for sample {idx}")
            # Fallback: try to get from class name if available
            try:
                class_name = meta_entry.get('label', '')
                make_label, type_label, class_label = self.hierarchy_utils.get_hierarchical_labels(class_name)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not get hierarchical labels for {class_name}: {e}")
                make_label, type_label, class_label = -1, -1, -1
        
        return image, (make_label, type_label, class_label)
    
def analyze_data_distribution(dataset, hierarchy_utils):
    """Analyze the distribution of classes in the dataset"""
    make_counts = defaultdict(int)
    type_counts = defaultdict(int)
    class_counts = defaultdict(int)
    unknown_count = 0
    
    print("Analyzing data distribution...")
    for i in tqdm(range(len(dataset)), desc="Analyzing samples", unit="sample"):
        _, (make_label, type_label, class_label) = dataset[i]
        if make_label != -1:  # Valid labels
            make_counts[make_label] += 1
            type_counts[type_label] += 1
            class_counts[class_label] += 1
        else:
            unknown_count += 1
    
    print("Data Distribution:")
    print(f"Make distribution: {dict(make_counts)}")
    print(f"Type distribution: {dict(type_counts)}")
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Unknown classes: {unknown_count}")
    
    # Check for class imbalance
    total_samples = sum(class_counts.values())
    print(f"\nTotal valid samples: {total_samples}")
    
    # Check if any class is dominant
    if class_counts:
        max_class_count = max(class_counts.values())
        max_ratio = max_class_count / total_samples
        print(f"Most frequent class ratio: {max_ratio:.2f}")
        
        if max_ratio > 0.8:
            print("WARNING: Severe class imbalance detected!")
        
        # Print the most frequent class details
        most_frequent_class_label = max(class_counts, key=class_counts.get)
        print(f"Most frequent class label: {most_frequent_class_label} with {max_class_count} samples")
    
    return make_counts, type_counts, class_counts

class HierarchicalModel(nn.Module):
    """Hierarchical classification model with separate heads for each level"""
    
    def __init__(self, input_channels=3, num_make_classes=3, num_type_classes=5, num_class_classes=27):
        super(HierarchicalModel, self).__init__()
        
        # Use ResNet18 as feature extractor
        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Feature dimension after CNN
        self.feature_dim = 512
        
        # Separate classification heads for each level
        self.make_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_make_classes)
        )
        
        self.type_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_type_classes)
        )
        
        self.class_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_class_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Get predictions from each head
        make_pred = self.make_classifier(features)
        type_pred = self.type_classifier(features)
        class_pred = self.class_classifier(features)
        
        return make_pred, type_pred, class_pred

class HierarchicalLoss(nn.Module):
    """Custom loss function for hierarchical classification"""
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(HierarchicalLoss, self).__init__()
        self.alpha = alpha  # Weight for make loss
        self.beta = beta    # Weight for type loss
        self.gamma = gamma  # Weight for class loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, predictions, targets):
        make_pred, type_pred, class_pred = predictions
        make_target, type_target, class_target = targets
        
        # Calculate individual losses
        make_loss = self.ce_loss(make_pred, make_target)
        type_loss = self.ce_loss(type_pred, type_target)
        class_loss = self.ce_loss(class_pred, class_target)
        
        # Combine losses with weights
        total_loss = (self.alpha * make_loss + 
                     self.beta * type_loss + 
                     self.gamma * class_loss)
        
        return total_loss, make_loss, type_loss, class_loss
    
def generate_confusion_matrices(model, test_loader, hierarchy_utils, device='cuda', save_dir='cm'):
    """Generate and save confusion matrices for all hierarchical levels"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_make_preds = []
    all_make_targets = []
    all_type_preds = []
    all_type_targets = []
    all_class_preds = []
    all_class_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Generating predictions", unit="batch"):
            images = images.to(device)
            make_targets = targets[0].to(device)
            type_targets = targets[1].to(device)
            class_targets = targets[2].to(device)
            
            # Get predictions
            make_pred, type_pred, class_pred = model(images)
            
            # Store predictions and targets
            all_make_preds.extend(make_pred.argmax(1).cpu().numpy())
            all_make_targets.extend(make_targets.cpu().numpy())
            
            all_type_preds.extend(type_pred.argmax(1).cpu().numpy())
            all_type_targets.extend(type_targets.cpu().numpy())
            
            all_class_preds.extend(class_pred.argmax(1).cpu().numpy())
            all_class_targets.extend(class_targets.cpu().numpy())
    
    # Get label mappings for confusion matrix labels
    make_labels = list(hierarchy_utils.level_mappings['modulation'].keys())
    type_labels = list(hierarchy_utils.level_mappings['protocol'].keys())
    class_labels = list(hierarchy_utils.level_mappings['label'].keys())

    # Get the number of classes defined in hierarchy
    num_make_classes = len(make_labels)
    num_type_classes = len(type_labels)
    num_class_classes = len(class_labels)
    
    # Create label indices for all possible classes
    make_label_indices = list(range(num_make_classes))
    type_label_indices = list(range(num_type_classes))
    class_label_indices = list(range(num_class_classes))
    
    # Generate confusion matrices with all possible labels
    make_cm = confusion_matrix(all_make_targets, all_make_preds, labels=make_label_indices)
    type_cm = confusion_matrix(all_type_targets, all_type_preds, labels=type_label_indices)
    class_cm = confusion_matrix(all_class_targets, all_class_preds, labels=class_label_indices)
    
    # Plot and save Make level confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(make_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=make_labels, yticklabels=make_labels)
    plt.title('Modulation Level Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'modulation_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot and save Type level confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(type_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=type_labels, yticklabels=type_labels)
    plt.title('Protocol Level Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'protocol_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot and save Class level confusion matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(class_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Model Level Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate and save classification reports with zero_division parameter and labels
    make_report = classification_report(
        all_make_targets, all_make_preds, 
        labels=make_label_indices,
        target_names=make_labels, 
        output_dict=True,
        zero_division=0
    )
    
    type_report = classification_report(
        all_type_targets, all_type_preds,
        labels=type_label_indices,
        target_names=type_labels, 
        output_dict=True,
        zero_division=0
    )
    
    class_report = classification_report(
        all_class_targets, all_class_preds,
        labels=class_label_indices,
        target_names=class_labels, 
        output_dict=True,
        zero_division=0
    )
    
    # Save classification reports as JSON
    with open(os.path.join(save_dir, 'modulation_classification_report.json'), 'w') as f:
        json.dump(make_report, f, indent=2)

    with open(os.path.join(save_dir, 'protocol_classification_report.json'), 'w') as f:
        json.dump(type_report, f, indent=2)

    with open(os.path.join(save_dir, 'model_classification_report.json'), 'w') as f:
        json.dump(class_report, f, indent=2)
    
    # Calculate accuracies manually to avoid KeyError
    make_accuracy = np.mean(np.array(all_make_targets) == np.array(all_make_preds))
    type_accuracy = np.mean(np.array(all_type_targets) == np.array(all_type_preds))
    class_accuracy = np.mean(np.array(all_class_targets) == np.array(all_class_preds))
    
    # Print summary statistics safely
    print(f"\n=== Confusion Matrices and Reports saved to '{save_dir}' folder ===")
    print(f"Make Level Accuracy: {make_accuracy:.4f}")
    print(f"Type Level Accuracy: {type_accuracy:.4f}")
    print(f"Class Level Accuracy: {class_accuracy:.4f}")
    
    # Print class distribution info
    print(f"\nDataset Info:")
    print(f"Unique classes present in data: {len(set(all_class_targets))}")
    print(f"Total classes in hierarchy: {num_class_classes}")
    print(f"Present class labels: {sorted(set(all_class_targets))}")
    
    # Print what keys are available in reports for debugging
    print(f"\nReport keys available:")
    print(f"Make report keys: {list(make_report.keys())}")
    
    return {
        'make_cm': make_cm,
        'type_cm': type_cm,
        'class_cm': class_cm,
        'make_report': make_report,
        'type_report': type_report,
        'class_report': class_report,
        'make_accuracy': make_accuracy,
        'type_accuracy': type_accuracy,
        'class_accuracy': class_accuracy
    }

def train_hierarchical_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """Training function for hierarchical model"""
    
    # Loss function and optimizer
    criterion = HierarchicalLoss(alpha=1.0, beta=1.0, gamma=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    model.to(device)
    
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        # Training phase
        model.train()
        train_loss = 0.0
        train_make_acc = 0.0
        train_type_acc = 0.0
        train_class_acc = 0.0

        # Add tqdm progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         leave=False, unit="batch")
        
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = images.to(device)
            make_targets = targets[0].to(device)
            type_targets = targets[1].to(device)
            class_targets = targets[2].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            make_pred, type_pred, class_pred = model(images)
            
            # Calculate loss
            total_loss, make_loss, type_loss, class_loss = criterion(
                (make_pred, type_pred, class_pred),
                (make_targets, type_targets, class_targets)
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Calculate accuracies
            with torch.no_grad():
                make_acc = (make_pred.argmax(1) == make_targets).float().mean()
                type_acc = (type_pred.argmax(1) == type_targets).float().mean()
                class_acc = (class_pred.argmax(1) == class_targets).float().mean()
                
                train_make_acc += make_acc.item()
                train_type_acc += type_acc.item()
                train_class_acc += class_acc.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_make_acc = 0.0
        val_type_acc = 0.0
        val_class_acc = 0.0
        
        # Add tqdm progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                       leave=False, unit="batch")

        with torch.no_grad():
            for images, targets in val_pbar:
                images = images.to(device)
                make_targets = targets[0].to(device)
                type_targets = targets[1].to(device)
                class_targets = targets[2].to(device)
                
                make_pred, type_pred, class_pred = model(images)
                
                total_loss, _, _, _ = criterion(
                    (make_pred, type_pred, class_pred),
                    (make_targets, type_targets, class_targets)
                )
                
                val_loss += total_loss.item()
                
                make_acc = (make_pred.argmax(1) == make_targets).float().mean()
                type_acc = (type_pred.argmax(1) == type_targets).float().mean()
                class_acc = (class_pred.argmax(1) == class_targets).float().mean()
                
                val_make_acc += make_acc.item()
                val_type_acc += type_acc.item()
                val_class_acc += class_acc.item()
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Modulation Acc: {train_make_acc/len(train_loader):.4f}, "
              f"Protocol Acc: {train_type_acc/len(train_loader):.4f}, "
              f"Model Acc: {train_class_acc/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Modulation Acc: {val_make_acc/len(val_loader):.4f}, "
              f"Protocol Acc: {val_type_acc/len(val_loader):.4f}, "
              f"Model Acc: {val_class_acc/len(val_loader):.4f}")

        scheduler.step()

def main():
    """Main function to run hierarchical classification"""
    
    # Initialize hierarchy utilities
    hierarchy_utils = HierarchyUtils(hierarchy)
    
    print("Hierarchy Mappings:")
    print("Modulation:", hierarchy_utils.level_mappings['modulation'])
    print("Protocol:", hierarchy_utils.level_mappings['protocol'])
    print("Model:", hierarchy_utils.level_mappings['label'])

    # Get number of classes for each level
    num_make, num_type, num_class = hierarchy_utils.get_num_classes_per_level()
    print(f"\nNumber of classes: Modulation={num_make}, Protocol={num_type}, Model={num_class}")

    # Path to your cleaned metadata JSON file
    cleaned_meta_data_path = '/mnt/d/OneDrive - Rowan University/RA/Summer 25/Raptor/hierarchy/cleaned_meta_data.json'
    
    # Create the original SpectrogramDataset
    original_dataset = SpectogramDataset(
        master_meta_data_path=cleaned_meta_data_path,
        snr_list=[None],  # No noise for now
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    )

    # Wrap with hierarchical functionality
    hierarchical_dataset = HierarchicalSpectrogramDataset(original_dataset, hierarchy_utils)
    
    # Split into train/val (you can adjust the split ratio)
    dataset_size = len(hierarchical_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size 

    train_dataset, val_dataset = torch.utils.data.random_split(hierarchical_dataset, [train_size, val_size])

    # Analyze data distribution
    print("\n=== ANALYZING DATA DISTRIBUTION ===")
    analyze_data_distribution(hierarchical_dataset, hierarchy_utils)

    # Check a few samples
    print("\n=== SAMPLE DATA CHECK ===")
    for i in range(min(5, len(hierarchical_dataset))):
        image, (make_label, type_label, class_label) = hierarchical_dataset[i]
        print(f"Sample {i}: Make={make_label}, Type={type_label}, Class={class_label}")
        print(f"Image shape: {image.shape}")
        
        # Also print the original metadata for verification
        meta = original_dataset.master_metadata[i]
        print(f"  Original label: {meta.get('label', 'N/A')}")
        print(f"  Make: {meta.get('make', 'N/A')}, Type: {meta.get('type', 'N/A')}")
        print("---")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = HierarchicalModel(
        input_channels=3,
        num_make_classes=num_make,
        num_type_classes=num_type,
        num_class_classes=num_class
    )
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_hierarchical_model(model, train_loader, val_loader, num_epochs=25, device=device)
    
    # Save model
    torch.save(model.state_dict(), 'hierarchical_model.pth')
    print("Model saved!")

    # Generate confusion matrices on validation set
    print("\nGenerating confusion matrices...")
    generate_confusion_matrices(model, val_loader, hierarchy_utils, device, save_dir='cm')
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()