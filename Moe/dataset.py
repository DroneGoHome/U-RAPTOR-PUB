from generator_data_loader import SpectogramDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.constants import FFT_SIZE, SAMPLING_RATE, NUM_FFT_SPEC
from typing import List

master_meta_path = '/mnt/d/Raptor/Moe_data/master_meta_data.json'

custom_dataset = SpectogramDataset(
    master_meta_data_path=master_meta_path,
    transform=transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),  # Normalize if needed
    fft_size=FFT_SIZE, # Pass relevant params
    num_fft_spec=1500, # from constants.NUM_FFT_SPEC
    stft_overlap=128,)

print("Dataset initialized.")
print(f"Number of samples in dataset: {len(custom_dataset)}")
print(f"Class mapping: {custom_dataset.label_to_int}")

if len(custom_dataset) > 0:
    print("\nFetching first sample directly from dataset:")

    print("\nTesting DataLoader...")
    try:
        data_loader = DataLoader(custom_dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers > 0 can speed up
        
        samples_batch, targets_batch = next(iter(data_loader))
        
        print("Fetched one batch from DataLoader:")
        print(f"Samples batch type: {type(samples_batch)}")
        print(f"Samples batch shape: {samples_batch.shape}")
        print(f"Samples batch dtype: {samples_batch.dtype}")

        print(f"Targets batch type: {type(targets_batch)}")
        print(f"Targets batch: {targets_batch}")
        print(f"Targets batch shape: {targets_batch.shape}")

    except Exception as e:
        print(f"Error during DataLoader test: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Dataset is empty, cannot test fetching or DataLoader.")
