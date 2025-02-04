import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import mean_absolute_error
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset2 import MedicalImageDatasetBalancedIntensity3D
from model import Backbone, SingleScanModel, Classifier
from utils import BaseConfig

class BrainAgeInference(BaseConfig):
    """
    Inference class for brain age prediction model.
    """

    def __init__(self):
        """Initialize the inference setup with model and data."""
        super().__init__()
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        config = self.get_config()
        self.backbone = Backbone()
        self.classifier = Classifier(d_model=2048)
        self.model = SingleScanModel(self.backbone, self.classifier)
        
        # Load weights
        checkpoint = torch.load(config["infer"]["checkpoints"], map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model and checkpoint loaded!")
        
    ## spinup dataloaders    
    def setup_data(self):
        config = self.get_config()
        self.test_dataset = MedicalImageDatasetBalancedIntensity3D(
            csv_path=config["data"]["test_csv"],
            root_dir=config["data"]["root_dir"]
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.custom_collate,
            num_workers=1
        )
            
    def infer(self):
        """ Infer pass """
        results_df = pd.DataFrame(columns=['PredictedAge', 'TrueAge'])
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for sample in tqdm(self.test_loader, desc="Inference", unit="batch"):
                inputs = sample['image'].to(self.device)
                labels = sample['label'].float().to(self.device)
                
                with autocast():
                    outputs = self.model(inputs)
                
                predictions = outputs.cpu().numpy().flatten()
                all_labels.extend(labels.cpu().numpy().flatten())
                all_predictions.extend(predictions)
                
                result = pd.DataFrame({
                    'PredictedAge': predictions,
                    'TrueAge': labels.cpu().numpy().flatten()
                })
                results_df = pd.concat([results_df, result], ignore_index=True)
        
        mae = mean_absolute_error(all_labels, all_predictions)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        results_df.to_csv('infer_output.csv', index=False)
        
        return mae

if __name__ == "__main__":
    inferencer = BrainAgeInference()
    mae = inferencer.infer()