import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset2 import MedicalImageDatasetBalancedIntensity3D
from model import Backbone, SingleScanModel, Classifier
from utils import BaseConfig



def calculate_metrics(pred_probs, pred_labels, true_labels):
    """
    Multi-class classification metrics.
    Args:
        pred_probs (numpy.ndarray): Predicted probabilities for each class
        pred_labels (numpy.ndarray): Predicted labels
        true_labels (numpy.ndarray): Ground truth labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1, and AUC metrics
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    # For multi-class, we use ROC AUC OVR (One-vs-Rest)
    auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


#============================
#  INFERENCE CLASS
#============================
class SequenceInference(BaseConfig):
    """
    Inference class for sequence classification model.
    """

    def __init__(self):
        super().__init__()
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        config = self.get_config()
        self.backbone = Backbone()
        self.classifier = Classifier(d_model=2048, num_classes=4)  # 4-way classification
        self.model = SingleScanModel(self.backbone, self.classifier)
        
        # Load weights
        checkpoint = torch.load(config["infer"]["checkpoints"], map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model and checkpoint loaded!")
        
    ## spin up data loaders
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
        """
        Run inference pass
        
        Returns:
            dict: Dictionary with evaluation metrics
        """
        results_df = pd.DataFrame(columns=['PredictedProbs_Class0', 'PredictedProbs_Class1', 
                                         'PredictedProbs_Class2', 'PredictedProbs_Class3',
                                         'PredictedLabel', 'TrueLabel'])
        all_labels = []
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for sample in tqdm(self.test_loader, desc="Inference", unit="batch"):
                inputs = sample['image'].to(self.device)
                labels = sample['label'].to(self.device)
                
                with autocast():
                    outputs = self.model(inputs)
                
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds)
                all_probs.extend(probs)
                
                result = pd.DataFrame({
                    'PredictedProbs_Class0': probs[:, 0],
                    'PredictedProbs_Class1': probs[:, 1],
                    'PredictedProbs_Class2': probs[:, 2],
                    'PredictedProbs_Class3': probs[:, 3],
                    'PredictedLabel': preds,
                    'TrueLabel': labels.cpu().numpy()
                })
                results_df = pd.concat([results_df, result], ignore_index=True)
        
        # log metrics 
        # metrics = calculate_metrics(
        #     np.array(all_probs),
        #     np.array(all_predictions),
        #     np.array(all_labels)
        # )
        
        # print("\nTest Set Metrics:")
        # print(f"Accuracy: {metrics['accuracy']:.4f}")
        # print(f"Precision: {metrics['precision']:.4f}")
        # print(f"Recall: {metrics['recall']:.4f}")
        # print(f"F1 Score: {metrics['f1']:.4f}")
        # print(f"AUC: {metrics['auc']:.4f}")
        
        # Save results
        print("PredictedLabel", preds)
        results_df.to_csv('./data/output/sequence_classification_predictions.csv', index=False)
        
        return None #metrics

if __name__ == "__main__":
    inferencer = SequenceInference()
    _ = inferencer.infer()