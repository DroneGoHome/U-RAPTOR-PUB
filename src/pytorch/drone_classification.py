# Python file to classify the ten drone images

import os
import json
import torch
from torchvision import transforms, models, datasets
from PIL import Image
import random
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import argparse
from generator_data_loader import SpectogramDataset

def get_resnet18_model(num_classes=10):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def save_json(data, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file_path}")
    except IOError as e:
        print(f"Error saving data: {e}")

def load_model(model_path, num_classes, device):
    """
    Load the trained model from a file.
    
    Args:
        model_path (str): Path to the saved model
        device (torch.device): Device to load the model on
    
    Returns:
        model: The loaded model
    """
    model = get_resnet18_model(num_classes=num_classes).to(device)
    
    # If model was saved as DataParallel, we need to handle that
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        # Model was saved using DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    return model

def main(args):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Use ImageFolder to load the dataset
    # full_dataset = datasets.ImageFolder(root=args.main_path, transform=train_transform)
    full_dataset = SpectogramDataset(master_meta_data_path=args.main_path, transform=train_transform, snr_list=[None, 10, 20],)
    args.num_classes = full_dataset.num_classes  # Update num_classes based on dataset
    
    # Save label mappings to a JSON file
    label_mappings = {
        "label_to_int": full_dataset.label_to_int,
        "int_to_label": full_dataset.int_to_label,
        "classes": full_dataset.classes
    }
    os.makedirs(args.cm_dir, exist_ok=True) # Ensure the output directory exists
    mappings_save_path = os.path.join(args.cm_dir, "label_mappings.json")
    save_json(label_mappings, mappings_save_path)


    # Calculate sizes for train/validation split
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.val_ratio)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms after splitting
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=23, pin_memory=True
    )
    
    test_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=23, pin_memory=True
    )

    print(f"Dataset classes: {full_dataset.classes}")
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Get model and move to device
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda")
        model =  load_model(model_path=args.save_path, num_classes=args.num_classes, device='cpu') if os.path.exists(args.save_path) else get_resnet18_model(num_classes=args.num_classes)
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        print(f"Using {torch.cuda.device_count()} GPUs.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_resnet18_model(num_classes=args.num_classes).to(device)
        print(f"Using device: {device}")

    # Store the class names for confusion matrix
    class_names = full_dataset.classes
    print(f"Class names: {class_names}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        print(f"Epoch {epoch+1} done. Avg Train Loss: {avg_train_loss:.4f}")

        # Evaluation after each epoch
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_batches = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1} [Eval]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        val_acc = 100 * correct / total if total > 0 else 0.0
        print(f"Validation Loss after epoch {epoch+1}: {avg_val_loss:.4f}")
        print(f"Validation Accuracy after epoch {epoch+1}: {val_acc:.2f}%")

        # Save model and confusion matrix if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Model saved at epoch {epoch+1} with validation loss {avg_val_loss:.4f}")

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(all_labels, all_preds),
                "precision": precision_score(all_labels, all_preds, average='weighted', zero_division=0),
                "recall": recall_score(all_labels, all_preds, average='weighted', zero_division=0),
                "f1": f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            }

            # Save metrics to a text file
            with open(os.path.join(args.cm_dir, "metrics.txt"), "w") as f:
                for metric_name, metric_value in metrics.items():
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
            # Save (replace) confusion matrix image for the best model
                    # Confusion Matrix
            labelsize = 16  # Adjusted for better visibility
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

            # Adjust figure size dynamically, ensure it's large enough for bold text
            fig_width = max(10, len(class_names) * 0.8) # Increased base width and scaling factor
            fig_height = max(10, len(class_names) * 0.8) # Increased base height and scaling factor
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

            disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', colorbar=False, text_kw={'fontsize': labelsize, 'fontweight': 'bold'}) # Adjust text_kw for cell text

            # Increase font size and make bold for x and y tick labels
            ax.tick_params(axis='x', labelsize=labelsize, labelfontfamily='sans-serif', labelcolor='black') # fontweight doesn't work directly here, use setp
            ax.tick_params(axis='y', labelsize=labelsize, labelfontfamily='sans-serif', labelcolor='black') # fontweight doesn't work directly here, use setp

            plt.setp(ax.get_xticklabels(), fontweight="bold")
            plt.setp(ax.get_yticklabels(), fontweight="bold")

            if len(class_names) > 6:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                plt.setp(ax.get_yticklabels(), rotation=0) # Keep y-axis labels horizontal or adjust as needed
            
            plt.tight_layout(pad=2.0) # Add some padding
            os.makedirs(args.cm_dir, exist_ok=True)
            cm_save_path = os.path.join(args.cm_dir, "confusion_matrix_best.png")
            plt.savefig(cm_save_path)
            plt.close(fig)
            plt.close('all')  # Ensure all figures are closed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 for binary classification")
    parser.add_argument('--main_path', type=str, default='/mnt/d/Raptor/binary/drones', 
                        help='Main data directory with subdirectories for each class')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/best_resnet18_drones.pth', help='Path to save best model')
    parser.add_argument('--cm_dir', type=str, default='cm_drones', help='Directory to save confusion matrix images')
    args = parser.parse_args()

    # Make sure the model directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    main(args)
