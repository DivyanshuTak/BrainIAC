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
from model import Backbone, SingleScanModel, Classifier
from utils import BaseConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(pred_probs, pred_labels, true_labels):
    """
    Multi-class classification metrics.
    
    Args:
        pred_probs (numpy.ndarray): Predicted probabilities for each class
        pred_labels (numpy.ndarray): Predicted labels
        true_labels (numpy.ndarray): Ground truth labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1, and AUC
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

#============================
#  TRAINER CLASS
#============================

class SequenceTrainer(BaseConfig):
    """
    Trainer class for sequence classification 
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
        # Change classifier to output 4 values for multi-class classification
        self.classifier = Classifier(d_model=2048, num_classes=4)
        self.model = SingleScanModel(self.backbone, self.classifier)
        
        # Load weights from brainiac
        config = self.get_config()
        if config["train"]["finetune"] == "yes":
            checkpoint = torch.load(config["train"]["weights"])
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
        self.train_dataset = TransformationMedicalImageDatasetBalancedIntensity3D(
            csv_path=config['data']['train_csv'],
            root_dir=config["data"]["root_dir"]
        )
        self.val_dataset = MedicalImageDatasetBalancedIntensity3D(
            csv_path=config['data']['val_csv'],
            root_dir=config["data"]["root_dir"]
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            collate_fn=self.custom_collate,
            num_workers=config["data"]["num_workers"]
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.custom_collate,
            num_workers=1
        )
        
    def setup_training(self):
        """
        training setup
        """
        config = self.get_config()
        # Cross Entropy Loss for multi-class classification
        self.criterion = nn.CrossEntropyLoss().to(self.device)
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
            
            # save model based on auc 
            if  metrics['auc'] > best_metrics['auc']:
                print(f"New best model found!")
                print(f"Improved Val Loss from {best_metrics['val_loss']:.4f} to {val_loss:.4f}")
                print(f"Improved F1 from {best_metrics['f1']:.4f} to {metrics['f1']:.4f}")
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
            labels = sample['label'].to(self.device)  # No need for float() conversion
            
            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)  # CrossEntropyLoss expects raw logits
            
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
                labels = sample['label'].to(self.device)  # No need for float() conversion
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)  # CrossEntropyLoss expects raw logits
                
                # Get probabilities and predictions for multi-class
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                val_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)
                
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
    
    ## save checkpoint
    def save_checkpoint(self, epoch, loss, metrics):
        config = self.get_config()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics
        }
        save_path = os.path.join(
            config['logger']['save_dir'],
            config['logger']['save_name'].format(epoch=epoch, loss=loss, metric=metrics['f1'])
        )
        torch.save(checkpoint, save_path)

if __name__ == "__main__":
    trainer = SequenceTrainer()
    trainer.train()