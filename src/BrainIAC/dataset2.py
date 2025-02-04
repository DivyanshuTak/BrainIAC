import os
import torch
import pandas as pd 
from torch.utils.data import Dataset
import nibabel as nib 
from monai.transforms import Affined, RandGaussianNoised, Rand3DElasticd, AdjustContrastd, ScaleIntensityd, ToTensord, Resized, RandRotate90d, Resize, RandGaussianSmoothd, GaussianSmoothd, Rotate90d, StdShiftIntensityd, RandAdjustContrastd, Flipd
import random 
import numpy as np



    
#######################################
##      3D SYNC TRANSFORM
#######################################

class NormalSynchronizedTransform3D:
    """ Vanilla transformation for inference"""
    
    def __init__(self, image_size=(128,128,128), max_rotation=40, translate_range=0.2, scale_range=(0.9, 1.3), apply_prob=0.5):
        self.image_size = image_size
        self.max_rotation = max_rotation
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.apply_prob = apply_prob

    def __call__(self, scan_list):
        transformed_scans = []
        resize_transform = Resized(spatial_size=(128,128,128), keys=["image"])
        scale_transform = ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)  # Intensity scaling
        tensor_transform = ToTensord(keys=["image"])  # Convert to tensor

        for scan in scan_list:
            sample = {"image": scan}
            sample = resize_transform(sample)
            sample = scale_transform(sample)
            sample = tensor_transform(sample)
            transformed_scans.append(sample["image"].squeeze())
            
        return torch.stack(transformed_scans)



class MedicalImageDatasetBalancedIntensity3D(Dataset):
    """ Dataset for loading images """

    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id":str, "scandate":str})
        self.root_dir = root_dir
        self.transform = NormalSynchronizedTransform3D()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        scan_dates = str(self.dataframe.loc[idx, 'scandate'])
        label = self.dataframe.loc[idx, 'label']
        scandates = scan_dates.split('-')
        scan_list = []


        for scandate in scandates:
            img_name = os.path.join(self.root_dir , f"{pat_id}_{scandate}.nii.gz") 
            if not os.path.exists(img_name):
                img_name = os.path.join(self.root_dir , f"{pat_id}_{scandate}.nii.gz") 
            scan = nib.load(img_name).get_fdata()
            scan_list.append(torch.tensor(scan, dtype=torch.float32).unsqueeze(0)) 

        transformed_scans = self.transform(scan_list)
        
         
        sample = {"image": transformed_scans, "label": torch.tensor(label, dtype=torch.float32), "pat_id": pat_id}  
        return sample

