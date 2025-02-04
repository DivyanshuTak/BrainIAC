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
    """ Vanilla Validation Transforms"""

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
    """ Validation Dataset class """

    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id":str, "scandate":str})
        self.root_dir = root_dir
        self.transform = NormalSynchronizedTransform3D()

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

         ## load the niftis from csv
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        #scan_dates = str(self.dataframe.loc[idx, 'scandate'])
        label = self.dataframe.loc[idx, 'label']
        #scandates = scan_dates.split('-')
        scan_list = []


        #for scandate in scandates:
        img_name = os.path.join(self.root_dir , f"{pat_id}.nii.gz") 
        if not os.path.exists(img_name):
            img_name = os.path.join(self.root_dir , f"{pat_id}.nii.gz") 
        scan = nib.load(img_name).get_fdata()
        scan_list.append(torch.tensor(scan, dtype=torch.float32).unsqueeze(0)) 

        ## package into a dictionary for val loader
        transformed_scans = self.transform(scan_list)
        sample = {"image": transformed_scans, "label": torch.tensor(label, dtype=torch.float32), "pat_id": pat_id}  
        return sample


class SynchronizedTransform3D:
    """ Trainign Augmentation method """

    def __init__(self, image_size=(128,128,128), max_rotation=0.34, translate_range=15, scale_range=(0.9, 1.3), apply_prob=0.5, gaussian_sigma_range=(0.25, 1.5), gaussian_noise_std_range=(0.05, 0.09)):
        self.image_size = image_size
        self.max_rotation = max_rotation
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.apply_prob = apply_prob
        self.gaussian_sigma_range = gaussian_sigma_range
        self.gaussian_noise_std_range = gaussian_noise_std_range

    def __call__(self, scan_list):
        transformed_scans = []
        rotate_params = (random.uniform(-self.max_rotation, self.max_rotation),) * 3 if random.random() < self.apply_prob else (0, 0, 0)
        translate_params = tuple([random.uniform(-self.translate_range, self.translate_range) for _ in range(3)]) if random.random() < self.apply_prob else (0, 0, 0)
        scale_params = tuple([random.uniform(self.scale_range[0], self.scale_range[1]) for _ in range(3)]) if random.random() < self.apply_prob else (1, 1, 1)
        gaussian_sigma = tuple([random.uniform(self.gaussian_sigma_range[0], self.gaussian_sigma_range[1]) for _ in range(3)]) if random.random() < self.apply_prob else None
        gaussian_noise_std = random.uniform(self.gaussian_noise_std_range[0], self.gaussian_noise_std_range[1]) if random.random() < self.apply_prob else None
        flip_axes = (0,1) if random.random() < self.apply_prob else None  # Determine if and along which axes to flip
        flip_x = 0 if random.random() < self.apply_prob else None
        flip_y = 1 if random.random() < self.apply_prob else None
        flip_z = 2 if random.random() < self.apply_prob else None
        offset = random.randint(50,100) if random.random() < self.apply_prob else None 
        gammafactor = random.uniform(0.5,2.0) if random.random() < self.apply_prob else 1 

        affine_transform = Affined(keys=["image"], rotate_params=rotate_params, translate_params=translate_params, scale_params=scale_params, padding_mode='zeros')
        gaussian_blur_transform = GaussianSmoothd(keys=["image"], sigma=gaussian_sigma) if gaussian_sigma else None
        gaussian_noise_transform = RandGaussianNoised(keys=["image"], std=gaussian_noise_std, prob=1.0, mean=0.0, sample_std=False) if gaussian_noise_std else None
        #flip_transform = Rotate90d(keys=["image"], k=1, spatial_axes=flip_axes) if flip_axes else None
        flip_x_transform = Flipd(keys=["image"], spatial_axis=flip_x) if flip_x else None
        flip_y_transform = Flipd(keys=["image"], spatial_axis=flip_y) if flip_y else None
        flip_z_transform = Flipd(keys=["image"], spatial_axis=flip_z) if flip_z else None
        resize_transform = Resized(spatial_size=(128,128,128), keys=["image"])
        scale_transform = ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)  # Intensity scaling
        tensor_transform = ToTensord(keys=["image"])  # Convert to tensor
        shift_intensity = StdShiftIntensityd(keys = ["image"], factor = offset, nonzero=True)
        adjust_contrast = AdjustContrastd(keys = ["image"], gamma = gammafactor)

        for scan in scan_list:
            sample = {"image": scan}
            sample = resize_transform(sample)
            sample = affine_transform(sample)
            if flip_x_transform:
                sample = flip_x_transform(sample)
            if flip_y_transform:
                sample = flip_y_transform(sample)
            if flip_z_transform:
                sample = flip_z_transform(sample)
            if gaussian_blur_transform:
                sample = gaussian_blur_transform(sample) 
            if offset:
                sample = shift_intensity(sample)
            sample = scale_transform(sample)
            sample = adjust_contrast(sample)
            if gaussian_noise_transform:
                sample = gaussian_noise_transform(sample)
            sample = tensor_transform(sample)
            transformed_scans.append(sample["image"].squeeze())
            
        return torch.stack(transformed_scans)


class TransformationMedicalImageDatasetBalancedIntensity3D(Dataset):
    """ Training Dataset class """

    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id":str, "scandate":str})
        self.root_dir = root_dir
        self.transform = SynchronizedTransform3D() # calls training augmentations 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## load the niftis from csv
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        #scan_dates = str(self.dataframe.loc[idx, 'scandate'])
        label = self.dataframe.loc[idx, 'label']
        #scandates = scan_dates.split('-')
        scan_list = []


        #for scandate in scandates:
        img_name = os.path.join(self.root_dir , f"{pat_id}.nii.gz") #f"{pat_id}_{scandate}.nii.gz") 
        if not os.path.exists(img_name):
            img_name = os.path.join(self.root_dir , f"{pat_id}.nii.gz") 
        scan = nib.load(img_name).get_fdata()
        scan_list.append(torch.tensor(scan, dtype=torch.float32).unsqueeze(0)) 

        # package into a monai type dictionary 
        transformed_scans = self.transform(scan_list) 
        sample = {"image": transformed_scans, "label": torch.tensor(label, dtype=torch.float32), "pat_id": pat_id}  
        return sample
