import os
import torch
from torchvision import transforms, models
from PIL import Image
import random
from datagen_pickle import create_data_loaders
from tqdm import tqdm
import argparse
import numpy as np

def get_resnet18_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
    return model

def compute_confusion_matrix(true_labels, predictions, num_classes=10):
    """
    Compute confusion matrix without using scikit-learn
    
    Args:
        true_labels: List of true labels
        predictions: List of predicted labels
        num_classes: Number of classes
        
    Returns:
        confusion_matrix: 2D array of shape (num_classes, num_classes)
    """
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(true_labels, predictions):
        confusion_matrix[t, p] += 1
    return confusion_matrix.numpy()

def save_confusion_matrix_to_file(cm, class_names, filepath):
    """
    Save confusion matrix to a text file
    
    Args:
        cm: Confusion matrix as numpy array
        class_names: List of class names
        filepath: Path to save the confusion matrix
    """
    with open(filepath, 'w') as f:
        f.write("Confusion Matrix\n\n")
        f.write("True \\ Predicted | " + " | ".join(class_names) + "\n")
        f.write("-" * (20 + 10 * len(class_names)) + "\n")
        
        for i, class_name in enumerate(class_names):
            row = [class_name.ljust(15)]
            for j in range(cm.shape[1]):
                row.append(str(cm[i, j]).rjust(8))
            f.write(" | ".join(row) + "\n")

def save_confusion_matrix_as_image(cm, class_names, filepath):
    """
    Save confusion matrix as an image without using matplotlib
    Uses PIL to create a simple visualization
    
    Args:
        cm: Confusion matrix as numpy array
        class_names: List of class names
        filepath: Path to save the confusion matrix image
    """
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont
    
    # Create image size based on matrix size
    cell_size = 50
    margin = 120
    img_width = margin + cell_size * cm.shape[1]
    img_height = margin + cell_size * cm.shape[0]
    img = PIL.Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = PIL.ImageDraw.Draw(img)
    
    try:
        # Try to load a font, use default if not available
        font = PIL.ImageFont.truetype("arial.ttf", 12)
    except:
        font = PIL.ImageFont.load_default()
    
    # Draw title
    draw.text((10, 10), f"Confusion Matrix", fill=(0, 0, 0), font=font)
    
    # Draw labels on y-axis (true labels)
    for i, label in enumerate(class_names):
        draw.text((10, margin + i * cell_size + cell_size//2), label, fill=(0, 0, 0), font=font)
    
    # Draw labels on x-axis (predicted labels)
    for i, label in enumerate(class_names):
        draw.text((margin + i * cell_size + cell_size//2 - 20, margin // 2), label, fill=(0, 0, 0), font=font)
    
    # Draw grid and values
    max_value = np.max(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Calculate cell color (darker blue for higher values)
            value_ratio = cm[i, j] / max_value if max_value > 0 else 0
            blue_intensity = 255 - int(200 * value_ratio)
            color = (240, 240, blue_intensity)
            
            # Draw rectangle for cell
            x0, y0 = margin + j * cell_size, margin + i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
            
            # Draw text for count
            text = str(cm[i, j])
            text_color = (0, 0, 0) if blue_intensity > 128 else (255, 255, 255)
            draw.text((x0 + cell_size//2 - 10, y0 + cell_size//2 - 5), text, fill=text_color, font=font)
    
    # Save image
    img.save(filepath)

def main(args):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_loader, test_loader = create_data_loaders(
        root_dir=args.main_path,
        batch_size=args.batch_size, 
        test_ratio=args.val_ratio,  # Changed from val_ratio to test_ratio to match datagen_pickle.py
        transform=None,  # Remove transform as we'll handle it in the model
        file_extension=args.file_ext
    )

    # Use all available GPUs if present
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda")
        model = torch.nn.DataParallel(get_resnet18_model())
        model = model.to(device)
        print(f"Using {torch.cuda.device_count()} GPUs.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_resnet18_model().to(device)
        print(f"Using device: {device}")

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

            # Save confusion matrix to text file
            cm = compute_confusion_matrix(all_labels, all_preds, num_classes=args.num_classes)
            os.makedirs(args.cm_dir, exist_ok=True)
            
            # Save as text file
            cm_txt_path = os.path.join(args.cm_dir, f"confusion_matrix_epoch_{epoch+1}.txt")
            save_confusion_matrix_to_file(cm, class_names, cm_txt_path)
            
            # Save as image file
            cm_img_path = os.path.join(args.cm_dir, f"confusion_matrix_epoch_{epoch+1}.png")
            save_confusion_matrix_as_image(cm, class_names, cm_img_path)
            
            # Also save as numpy file for later processing
            np_save_path = os.path.join(args.cm_dir, f"confusion_matrix_epoch_{epoch+1}.npy")
            np.save(np_save_path, cm)
            
            print(f"Confusion Matrix saved to {cm_img_path} and {cm_txt_path}")
            
            # Only print detailed matrix for binary classification
            if args.num_classes == 2:
                print(f"Confusion Matrix for epoch {epoch+1}:")
                print("True \\ Predicted | no_drone | drones")
                print("-" * 40)
                print(f"no_drone         | {cm[0,0]:^8} | {cm[0,1]:^6}")
                print(f"drones           | {cm[1,0]:^8} | {cm[1,1]:^6}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 for binary classification")
    parser.add_argument('--main_path', type=str, default='/mnt/d/Raptor/binary/drones', help='Main data directory')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')  # Changed default from 10 to 2
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/best_resnet18_drones.pth', help='Path to save best model')
    parser.add_argument('--cm_dir', type=str, default='cm_pickles', help='Directory to save confusion matrix images')
    parser.add_argument('--file_ext', type=str, default='.pkl', help='Data file extension (.pkl)')
    args = parser.parse_args()

    # Create class names based on number of classes
    if args.num_classes == 10:
        class_names = [f"class_{i}" for i in range(args.num_classes)]
    else:
        # Default to binary classification names
        class_names = ["no_drone", "drones"]
    
    main(args)
