import os
import torch
from torchvision import transforms, models
from PIL import Image
import random
from datagen import get_dataloaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse

def get_resnet18_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
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

    train_loader, test_loader = get_dataloaders(
        args.main_path, args.class0_subpath, args.class1_subpath, args.num_samples,
        batch_size=args.batch_size, val_ratio=args.val_ratio,
        train_transform=train_transform, test_transform=test_transform
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

            # Save (replace) confusion matrix image for the best model
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no_drone", "drones"])
            fig, ax = plt.subplots(figsize=(5, 5))
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            plt.title(f'Confusion Matrix (Epoch {epoch+1})')
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", style="oblique")
            plt.tight_layout()
            os.makedirs(args.cm_dir, exist_ok=True)
            cm_save_path = os.path.join(args.cm_dir, "confusion_matrix_best.png")
            plt.savefig(cm_save_path)
            plt.close(fig)
            plt.close('all')  # Ensure all figures are closed

if __name__ == "__main__":
    print('test')
    parser = argparse.ArgumentParser(description="Train ResNet18 for binary classification")
    parser.add_argument('--main_path', type=str, default='/mnt/d/Raptor/binary', help='Main data directory')
    parser.add_argument('--class0_subpath', type=str, default='no_drone', help='Subdirectory for class 0')
    parser.add_argument('--class1_subpath', type=str, default='drones', help='Subdirectory for class 1')
    parser.add_argument('--num_samples', type=int, default=650, help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/best_resnet18_binary.pth', help='Path to save best model')
    parser.add_argument('--cm_dir', type=str, default='cm', help='Directory to save confusion matrix images')
    args = parser.parse_args()
    print(args)
    raise NotImplementedError("This script is not implemented yet.")
    
    main(args)
