import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import sys
import nibabel as nib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset2 import MedicalImageDatasetBalancedIntensity3D
from model import Backbone, SingleScanModelBP, Classifier
from utils import BaseConfig
from .os_utils import plot_km_curve, calculate_metrics


#============================
#  CUSTOM DATASET TO LOAD T1CE AND FLAIR PER SCAN 
#============================
class OSDataset(MedicalImageDatasetBalancedIntensity3D):
    """Dataset class for OS prediction that loads T1CE, FLAIR, T1 and T2 modalities"""
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the basic info from csv
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'label']
        scan_list = []
        
        # Load T1, T2, T1CE, FLAIR modalities
        modalities = ["t1n", "t2w", "t1c", "t2f"]

        for modality in modalities:
            img_name = os.path.join(self.root_dir, f"{pat_id}_{modality}.nii.gz")
            scan = nib.load(img_name).get_fdata()
            scan_list.append(torch.tensor(scan, dtype=torch.float32).unsqueeze(0))


        # Use the same transform from parent class
        transformed_scans = self.transform(scan_list)
        sample = {
            "image": transformed_scans,
            "label": torch.tensor(label, dtype=torch.float32),
            "pat_id": pat_id
        }
        return sample


#============================
#  INFERENCE CLASS
#============================
class OSInference(BaseConfig):
    """
    Inference class for OS model.
    """

    def __init__(self):
        super().__init__()
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        config = self.get_config()
        self.backbone = Backbone()
        self.classifier = Classifier(d_model=2048, num_classes=1)  # Binary classification
        self.model = SingleScanModelBP(self.backbone, self.classifier)
        
        # Load weights
        checkpoint = torch.load(config["infer"]["checkpoints"], map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model and checkpoint loaded!")
        
    ## spin up data loaders
    def setup_data(self):
        config = self.get_config()
        self.test_dataset = OSDataset(
            csv_path=config["data"]["test_csv"],
            root_dir=config["data"]["root_dir"]
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.custom_collate,   ## set custom collate to 4 for including t1ce, flair, t1 and t2
            num_workers=1
        )
            
    def infer(self):
        """
        Run inference pass for test set
        Returns:
            dict: Dictionary with evaluation metrics
        """
        
        results_df = pd.DataFrame(columns=['PredictedProb', 'PredictedLabel', 'PredictedRisk', 'TrueLabel', 'pat_id'])
        all_labels = []
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for sample in tqdm(self.test_loader, desc=f"Inference on test data", unit="batch"):
                inputs = sample['image'].to(self.device)
                labels = sample['label'].float().to(self.device)
                pat_ids = sample['pat_id']
                with autocast():
                    outputs = self.model(inputs)
                
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                all_labels.extend(labels.cpu().numpy().flatten())
                all_predictions.extend(preds)
                all_probs.extend(probs)
                
                result = pd.DataFrame({
                    'PredictedProb': probs,
                    'PredictedLabel': preds,
                    'PredictedRisk': 1.0 - probs,
                    'TrueLabel': labels.cpu().numpy().flatten(),
                    'pat_id': pat_ids
                })
                results_df = pd.concat([results_df, result], ignore_index=True)

        # log metrics 
        metrics = calculate_metrics(
            np.array(all_probs),
            np.array(all_predictions),
            np.array(all_labels)
        )
        
        # print("\nTest Set Metrics:")
        # print(f"Accuracy: {metrics['accuracy']:.4f}")
        # print(f"Precision: {metrics['precision']:.4f}")
        # print(f"Recall: {metrics['recall']:.4f}")
        # print(f"F1 Score: {metrics['f1']:.4f}")
        # print(f"AUC: {metrics['auc']:.4f}")
        
        print("PredictedLabel", results_df["PredictedLabel"][0])

        # Save results
        results_df.drop(columns =["PredictedProb", "TrueLabel"] ).to_csv('./data/output/overall_survival_predictions.csv', index=False)
   
        # Uncomment if clinical data is available to conduct KM analysis on test set
        # clinical_df = pd.read_csv("../clinical/clinical_filtered.csv")
        # clinical_df = clinical_df[["pat_id", "survival_years", "deadstatus_event", "survival"]]         
        # merged_results = results_df.merge(clinical_df,how = "left")[["pat_id", "PredictedRisk", "survival_years","deadstatus_event", "survival"]]
        # split_median = 0.5215  # median threshold calculated using validation data      
        # merged_results['group'] = merged_results["PredictedRisk"].apply(lambda x: 'Low Risk' if x < split_median  else 'High Risk')
        # _ = plot_km_curve(merged_results)


if __name__ == "__main__":
    inferencer = OSInference()
    metrics = inferencer.infer()