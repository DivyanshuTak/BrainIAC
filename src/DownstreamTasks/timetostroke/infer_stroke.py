import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lifelines.utils import concordance_index
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset2 import MedicalImageDatasetBalancedIntensity3D
from model import Backbone, SingleScanModel, Classifier
from utils import BaseConfig



def calculate_metrics(pred, target):
    """
    Calculate evaluation metrics.
    Args:
        pred (numpy.ndarray): Model predictions
        target (numpy.ndarray): Ground truth values
        
    Returns:
        dict: Dictionary containing MAE, RMSE, R2, and C-index metrics
    """
    mae = mean_absolute_error(target, pred)
    rmse = np.sqrt(mean_squared_error(target, pred))
    r2 = r2_score(target, pred)
    try:
        c_index = concordance_index(target, pred, np.ones_like(target))
    except:
        c_index = 0
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'c_index': c_index}

class StrokeTimeInference(BaseConfig):
    """
    Inference class for time to stroke prediction model.
    """

    def __init__(self):
        super().__init__()
        self.setup_model()
        self.setup_data()
        
    ## setup model
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
        
    def setup_data(self):
        """
        Set up data loader for inference.
        """
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
        """
        Run inference pass
        
        Returns:
            dict: Dictionary containing  evaluation metrics
        """
        results_df = pd.DataFrame(columns=['PredictedTime', 'TrueTime'])
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
                    'PredictedTime': predictions,
                    'TrueTime': labels.cpu().numpy().flatten()
                })
                results_df = pd.concat([results_df, result], ignore_index=True)
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(np.array(all_predictions), np.array(all_labels))
        
        # Print detailed metrics
        print("\nTest Set Metrics:")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Concordance Index: {metrics['c_index']:.4f}")
        
        # Save results with additional statistics
        results_df['AbsoluteError'] = abs(results_df['PredictedTime'] - results_df['TrueTime'])
        stats = {
            'Metrics': metrics,
            'Statistics': {
                'Mean_Prediction': np.mean(all_predictions),
                'Std_Prediction': np.std(all_predictions),
                'Mean_True': np.mean(all_labels),
                'Std_True': np.std(all_labels)
            }
        }
        
        # Save results
        results_df.to_csv('stroke_time_predictions.csv', index=False)
        ## save the output metrrics 
        #pd.DataFrame([stats]).to_json('stroke_time_metrics.json')
        
        return metrics

if __name__ == "__main__":
    inferencer = StrokeTimeInference()
    metrics = inferencer.infer()