import torch
import numpy as np
import pandas as pd
import random
import yaml
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset2 import MedicalImageDatasetBalancedIntensity3D
from load_brainiac import load_brainiac

# fix random seed 
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Define custom collate function for data loading
def custom_collate(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]

    max_len = 1
    padded_images = []

    for img in images:
        pad_size = max_len - img.shape[0]
        if pad_size > 0:
            padding = torch.zeros((pad_size,) + img.shape[1:])
            img_padded = torch.cat([img, padding], dim=0)
            padded_images.append(img_padded)
        else:
            padded_images.append(img)

    return {"image": torch.stack(padded_images, dim=0), "label": torch.stack(labels)}

#=========================
# Inference function
#=========================

def infer(model, test_loader):
    features_df = None  # Placeholder for feature DataFrame
    model.eval()
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Inference", unit="batch"):
            inputs = sample['image'].to(device)
            class_labels = sample['label'].float().to(device)

            # Get features from the model
            features = model(inputs)
            features_numpy = features.cpu().numpy()

            # Expand features into separate columns
            feature_columns = [f'Feature_{i}' for i in range(features_numpy.shape[1])]
            batch_features = pd.DataFrame(
                features_numpy,
                columns=feature_columns
            )
            batch_features['GroundTruthClassLabel'] = class_labels.cpu().numpy().flatten()

            # Append batch features to features_df
            if features_df is None:
                features_df = batch_features
            else:
                features_df = pd.concat([features_df, batch_features], ignore_index=True)
    
    return features_df

#=========================
# Main inference pipeline
#=========================

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Extract BrainIAC features from images')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the BrainIAC model checkpoint')
    parser.add_argument('--input_csv', type=str, required=True,
                      help='Path to the input CSV file containing image paths')
    parser.add_argument('--output_csv', type=str, required=True,
                      help='Path to save the output features CSV')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing the image data')
    args = parser.parse_args()

    # spinup the dataloader
    test_dataset = MedicalImageDatasetBalancedIntensity3D(
        csv_path=args.input_csv,
        root_dir=args.root_dir
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=1
    )

    # Load brainiac
    model = load_brainiac(args.checkpoint, device)
    model = model.to(device)
    # infer
    features_df = infer(model, test_loader)
    
    # Save features
    features_df.to_csv(args.output_csv, index=False)
    print(f"Features saved to {args.output_csv}")

if __name__ == "__main__":
    main()