import os
import torch
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from other files
from binary_classifier import get_resnet18_model
from datagen import get_dataloaders

def load_model(model_path, device):
    """
    Load the trained model from a file.
    
    Args:
        model_path (str): Path to the saved model
        device (torch.device): Device to load the model on
    
    Returns:
        model: The loaded model
    """
    model = get_resnet18_model().to(device)
    
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

def evaluate_model(model, test_loader, device, output_dir=None):
    """
    Evaluate the model on test data
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        output_dir (str, optional): Directory to save evaluation results
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probability for class 1
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds)
    }
    
    # Create and save confusion matrix if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no_drone", "drones"])
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close(fig)
        
        # Print metrics
        print(f"\nEvaluation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
        
        # Save metrics to a text file
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    return metrics

def predict_image(model, image_path, transform, device):
    """
    Make a prediction for a single image
    
    Args:
        model: The trained model
        image_path (str): Path to the image
        transform: Transforms to apply to the image
        device: Device to run inference on
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_names = ["no_drone", "drones"]
    return class_names[predicted_class], confidence

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model = load_model(args.model_path, device)
    print(f"Model loaded from {args.model_path}")
    
    # Define transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Handle different inference modes
    if args.image_path:
        # Single image prediction
        class_name, confidence = predict_image(model, args.image_path, test_transform, device)
        print(f"Prediction: {class_name} with confidence {confidence:.4f}")
    
    else:
        # Evaluate on test data
        _, test_loader = get_dataloaders(
            args.main_path, args.class0_subpath, args.class1_subpath, args.num_samples,
            batch_size=args.batch_size, val_ratio=args.val_ratio,
            train_transform=test_transform, test_transform=test_transform
        )
        
        metrics = evaluate_model(model, test_loader, device, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with trained binary classification model")
    parser.add_argument('--model_path', type=str, default='models/best_resnet18_binary.pth', help='Path to the saved model')
    parser.add_argument('--main_path', type=str, default='/mnt/d/Raptor/binary', help='Main data directory')
    parser.add_argument('--class0_subpath', type=str, default='no_drone', help='Subdirectory for class 0')
    parser.add_argument('--class1_subpath', type=str, default='drones', help='Subdirectory for class 1')
    parser.add_argument('--num_samples', type=int, default=650, help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to save evaluation results')
    parser.add_argument('--image_path', type=str, default=None, help='Path to a single image for inference')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference if available')
    
    args = parser.parse_args()
    
    main(args)
