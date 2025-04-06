import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm 
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset2 import MedicalImageDatasetBalancedIntensity3D, TransformationMedicalImageDatasetBalancedIntensity3D
from model import Backbone, SingleScanModel, Classifier, SingleScanModelBP
from .os_utils import BaseConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import nibabel as nib




#============================
# CUSTOM DATASET CLASSES TO LAOD FOUR MODALITIES
#============================

class OSDatasetVal(MedicalImageDatasetBalancedIntensity3D):
    """Dataset class for OS prediction that loads T1CE, FLAIR, T1 and T2 modalities"""
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the basic info from csv
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'label']
        scan_list = []
        # Load T1CE, FLAIR, T1 and T2 modalities
        modalities = ["T1", "T2", "T1GD", "FLAIR"]
        data_dir = os.path.join(self.root_dir, "UPENN-GBM", "data")

        for modality in modalities:
            img_name = os.path.join(data_dir, pat_id, f"{pat_id}_{modality}.nii.gz")
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
    
class OSDatasetTrain(TransformationMedicalImageDatasetBalancedIntensity3D):
    """Dataset class for OS prediction that loads T1CE, FLAIR, T1 and T2 modalities"""
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the basic info from csv
        pat_id = str(self.dataframe.loc[idx, 'pat_id'])
        label = self.dataframe.loc[idx, 'survival']
        scan_list = []

        # Load T1CE, FLAIR, T1 and T2 modalities
        modalities = ["T1", "T2", "T1GD", "FLAIR"]
        data_dir = os.path.join(self.root_dir, "UPENN-GBM", "data")

        for modality in modalities:
            img_name = os.path.join(data_dir, pat_id, f"{pat_id}_{modality}.nii.gz")
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
#  TRAINER CLASS
#============================

class OSTrainer(BaseConfig):
    """
    trainer class for OS prediction
    """

    def __init__(self):
        super().__init__()
        self.setup_wandb()
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
    def setup_wandb(self):
        config = self.get_config()
        wandb.init(
            project=config['logger']['project_name'],
            name=config['logger']['run_name'],
            config=config
        )
        
    def setup_model(self):
        self.backbone = Backbone()  
        self.classifier = Classifier(d_model=2048)
        self.model = SingleScanModelBP(self.backbone, self.classifier)
        
        # Load weights from brainiac
        config = self.get_config()
        if config["train"]["finetune"] == "yes":
            checkpoint = torch.load(config["train"]["weights"], map_location=self.device)
            state_dict = checkpoint["state_dict"]
            filtered_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "backbone.") if key.startswith("module.") else key
                filtered_state_dict[new_key] = value
            self.model.backbone.load_state_dict(filtered_state_dict, strict=False)
            print("Pretrained weights loaded!")

        if config["train"]["freeze"] == "yes":
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("Backbone weights frozen!")
            
        self.model = self.model.to(self.device)
        
    ## spinup dataloaders
    def setup_data(self):
        config = self.get_config()
        self.train_dataset = OSDatasetTrain(
            csv_path=config['data']['train_csv'],
            root_dir=config["data"]["root_dir"]
        )
        self.val_dataset = OSDatasetVal(
            csv_path=config['data']['val_csv'],
            root_dir=config["data"]["root_dir"]
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"]
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        
    def setup_training(self):
        """
        training setup
        """
        config = self.get_config()
        # BCE loss  
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['optim']['lr'],
            weight_decay=config["optim"]["weight_decay"]
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['optim']['lr'],
            epochs=config['optim']['max_epochs'],
            steps_per_epoch=len(self.train_loader)
        )
        self.scaler = GradScaler()
    
    ## main training loop
    def train(self):
        config = self.get_config()
        max_epochs = config['optim']['max_epochs']
        best_metrics = {
            'val_loss': float('inf'),
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'auc': 0
        }

        for epoch in range(max_epochs):
            train_loss = self.train_epoch(epoch, max_epochs)
            val_loss, metrics = self.validate_epoch(epoch, max_epochs)
            
            # Save best model based on validation loss and AUC score
            if  metrics['auc'] > best_metrics['auc']:
                print(f"New best model found!")
                print(f"Improved Val Loss from {best_metrics['val_loss']:.4f} to {val_loss:.4f}")
                print(f"Improved AUC from {best_metrics['auc']:.4f} to {metrics['auc']:.4f}")
                best_metrics.update(metrics)
                best_metrics['val_loss'] = val_loss
                self.save_checkpoint(epoch, val_loss, metrics)
                
        wandb.finish()

    ## training pass
    def train_epoch(self, epoch, max_epochs):
        self.model.train()
        train_loss = 0.0
        
        for sample in tqdm(self.train_loader, desc=f"Training Epoch {epoch}/{max_epochs-1}"):
            inputs = sample['image'].to(self.device)
            labels = sample['label'].float().to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
            
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(self.train_loader.dataset)
        wandb.log({"Train Loss": train_loss})
        return train_loss
    
    ## validation pass
    def validate_epoch(self, epoch, max_epochs):
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for sample in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}/{max_epochs-1}"):
                inputs = sample['image'].to(self.device)
                labels = sample['label'].float().to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                
                # Get probabilities and predictions
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                val_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.flatten())
                all_probs.extend(probs.flatten())
                
        val_loss = val_loss / len(self.val_loader.dataset)
        metrics = calculate_metrics(
            np.array(all_probs),
            np.array(all_preds),
            np.array(all_labels)
        )
        
        wandb.log({
            "Val Loss": val_loss,
            "Accuracy": metrics['accuracy'],
            "Precision": metrics['precision'],
            "Recall": metrics['recall'],
            "F1 Score": metrics['f1'],
            "AUC": metrics['auc']
        })
        
        print(f"Epoch {epoch}/{max_epochs-1}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        
        return val_loss, metrics
    
    ## save best model
    def save_checkpoint(self, epoch, loss, metrics):
        config = self.get_config()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics
        }
        save_path = os.path.join(
            config['logger']['save_dir'],
            config['logger']['save_name'].format(epoch=epoch, loss=loss, metric=metrics['auc'])
        )
        torch.save(checkpoint, save_path)
        
def calculate_metrics(pred_probs, pred_labels, true_labels):
    """
    classification metrics.
    Args:
        pred_probs (numpy.ndarray): Predicted probabilities
        pred_labels (numpy.ndarray): Predicted labels
        true_labels (numpy.ndarray): Ground truth labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1, and AUC
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, pred_probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

if __name__ == "__main__":
    trainer = OSTrainer()
    trainer.train()