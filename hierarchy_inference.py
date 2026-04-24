import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from Moe.src.constants import hierarchy
from Moe.generator_data_loader import SpectogramDataset
from hierarchy_updated import HierarchyUtils, HierarchicalSpectrogramDataset, HierarchicalModel
from tqdm import tqdm

def load_model(model_path, num_make, num_type, num_class, device):
    model = HierarchicalModel(
        input_channels=3,
        num_make_classes=num_make,
        num_type_classes=num_type,
        num_class_classes=num_class
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    # Setup
    cleaned_meta_data_path = '/mnt/d/OneDrive - Rowan University/RA/Summer 25/Raptor/hierarchy/cleaned_meta_data.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hierarchy_utils = HierarchyUtils(hierarchy)
    num_make, num_type, num_class = hierarchy_utils.get_num_classes_per_level()

    # Dataset and split
    original_dataset = SpectogramDataset(
        master_meta_data_path=cleaned_meta_data_path,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    )
    hierarchical_dataset = HierarchicalSpectrogramDataset(original_dataset, hierarchy_utils)
    dataset_size = len(hierarchical_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    _, val_dataset = torch.utils.data.random_split(hierarchical_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load model
    model = load_model('hierarchical_model.pth', num_make, num_type, num_class, device)

    # Inference
    results = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Inference"):
            images = images.to(device)
            make_pred, type_pred, class_pred = model(images)
            make_pred_labels = make_pred.argmax(1).cpu().numpy()
            type_pred_labels = type_pred.argmax(1).cpu().numpy()
            class_pred_labels = class_pred.argmax(1).cpu().numpy()
            make_targets = targets[0].cpu().numpy()
            type_targets = targets[1].cpu().numpy()
            class_targets = targets[2].cpu().numpy()
            for i in range(len(images)):
                results.append({
                    'make_true': make_targets[i],
                    'type_true': type_targets[i],
                    'class_true': class_targets[i],
                    'make_pred': make_pred_labels[i],
                    'type_pred': type_pred_labels[i],
                    'class_pred': class_pred_labels[i]
                })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('hierarchical_inference_results.csv', index=False)

    # Compute and print accuracy for each level
    make_acc = (df['make_true'] == df['make_pred']).mean()
    type_acc = (df['type_true'] == df['type_pred']).mean()
    class_acc = (df['class_true'] == df['class_pred']).mean()
    print(f"Make accuracy: {make_acc:.4f}")
    print(f"Type accuracy: {type_acc:.4f}")
    print(f"Class accuracy: {class_acc:.4f}")
    print("Inference complete! Results saved to hierarchical_inference_results.csv")

if __name__ == "__main__":
    main()