import torch
import numpy as np
import random
import yaml
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import nibabel as nib
from monai.visualize.gradient_based import SmoothGrad, GuidedBackpropSmoothGrad
from dataset2 import MedicalImageDatasetBalancedIntensity3D
from load_brainiac import load_brainiac

# Fix random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# collate funcntion (unneccerary for single timpoint input)
def custom_collate(batch):
    """Handles variable size of the scans and pads the sequence dimension."""
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    patids = [item['pat_id'] for item in batch]
    
    max_len = 1  # singlescan input
    padded_images = []
    
    for img in images:
        pad_size = max_len - img.shape[0]
        if pad_size > 0:
            padding = torch.zeros((pad_size,) + img.shape[1:])
            img_padded = torch.cat([img, padding], dim=0)
            padded_images.append(img_padded)
        else:
            padded_images.append(img)

    return {"image": torch.stack(padded_images, dim=0), "label": labels, "pat_id": patids}


def generate_saliency_maps(model, data_loader, output_dir):
    """Generate saliency maps using guided backprop method"""
    model.eval()
    visualizer = GuidedBackpropSmoothGrad(model=model.backbone, stdev_spread=0.15, n_samples=10, magnitude=True)
    
    for sample in tqdm(data_loader, desc="Generating saliency maps"):
        inputs = sample['image'].requires_grad_(True)
        patids = sample["pat_id"]
        imagename = patids[0]
        
        input_tensor = inputs.to(device)
        
        with torch.enable_grad():
            saliency_map = visualizer(input_tensor)
        
        # Save input image and saliency map
        inputs_np = input_tensor.squeeze().cpu().detach().numpy()
        saliency_np = saliency_map.squeeze().cpu().detach().numpy()
        
        input_nifti = nib.Nifti1Image(inputs_np, np.eye(4))
        saliency_nifti = nib.Nifti1Image(saliency_np, np.eye(4))
        
        # Save files
        nib.save(input_nifti, os.path.join(output_dir, f"{imagename}_image.nii.gz"))
        nib.save(saliency_nifti, os.path.join(output_dir, f"{imagename}_saliencymap.nii.gz"))

def main():
    parser = argparse.ArgumentParser(description='Generate saliency maps for medical images')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--input_csv', type=str, required=True,
                      help='Path to the input CSV file containing image paths')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save saliency maps')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing the image data')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = MedicalImageDatasetBalancedIntensity3D(
        csv_path=args.input_csv,
        root_dir=args.root_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=1
    )
    
    # Load brainiac
    model = load_brainiac(args.checkpoint, device)
    model = model.to(device)
    
    # Generate saliency maps
    generate_saliency_maps(model, dataloader, args.output_dir)
    
    print(f"Saliency maps generated and saved to {args.output_dir}")

if __name__ == "__main__":
    main()