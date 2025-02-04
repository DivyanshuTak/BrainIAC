import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm 
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_absolute_error
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset2 import MedicalImageDatasetBalancedIntensity3D, TransformationMedicalImageDatasetBalancedIntensity3D
from model import Backbone, SingleScanModel, Classifier
from utils import BaseConfig
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from lifelines.utils import concordance_index


#============================
#  CUSTOM LOSS FUNCTION 
#============================

class CombinedLoss(nn.Module):
    """
    Combined loss function using MSE, MAE, and Huber losses.
    
    Args:
        alpha (float): Weight for MSE loss (default: 0.5)
        delta (float): Delta parameter for Huber loss (default: 1.0)
    """
    def __init__(self, alpha=0.5, delta=1.0):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.HuberLoss(delta=delta)
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        huber_loss = self.huber(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * (0.5 * mae_loss + 0.5 * huber_loss)

def calculate_metrics(pred, target):
    """
    Calculate evaluation metrics.
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

class StrokeTimeTrainer(BaseConfig):
    """
    A trainer class time to stroke prediction training
    """

    def __init__(self):
        """Initialize the trainer with model, data, and training pass."""
        super().__init__()
        self.setup_wandb()
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
    ## setup wandb logger    
    def setup_wandb(self):
        config = self.get_config()
        wandb.init(
            project=config['logger']['project_name'],
            name=config['logger']['run_name'],
            config=config
        )
        
    ## spin up the model     
    def setup_model(self):
        self.backbone = Backbone()
        self.classifier = Classifier(d_model=2048)
        self.model = SingleScanModel(self.backbone, self.classifier)
        
        # Load BrainIACs weights 
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

        # Freeze backbone if specified
        if config["train"]["freeze"] == "yes":
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("Backbone weights frozen!")
            
        self.model = self.model.to(self.device)
        
    def setup_data(self):
        """
        Data loaders for training and validation
        """
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
        Set up training components with combined loss and AdamW optimizer.
        """
        config = self.get_config()
        self.criterion = CombinedLoss(alpha=0.6, delta=1.0).to(self.device)
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
        
    def train(self):
        """
        Main training loop with enhanced metrics tracking.
        """
        config = self.get_config()
        max_epochs = config['optim']['max_epochs']
        best_metrics = {
            'val_loss': float('inf'),
            'mae': float('inf'),
            'rmse': float('inf'),
            'r2': -float('inf'),
            'c_index': -float('inf')
        }

        for epoch in range(max_epochs):
            train_loss = self.train_epoch(epoch, max_epochs)
            val_loss, metrics = self.validate_epoch(epoch, max_epochs)
            
            # Save best model based on multiple metrics
            if metrics['mae'] < best_metrics['mae'] and val_loss < best_metrics['val_loss']:
                print(f"New best model found!")
                print(f"Improved Val Loss from {best_metrics['val_loss']:.4f} to {val_loss:.4f}")
                print(f"Improved MAE from {best_metrics['mae']:.4f} to {metrics['mae']:.4f}")
                best_metrics.update({
                    'val_loss': val_loss,
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                    'c_index': metrics['c_index']
                })
                self.save_checkpoint(epoch, val_loss, metrics)
                
        wandb.finish()
                
    def train_epoch(self, epoch, max_epochs):
        """
        Training pass
        """
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
            
            # Add gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(self.train_loader.dataset)
        wandb.log({"Train Loss": train_loss})
        return train_loss
        
    def validate_epoch(self, epoch, max_epochs):
        """
        Validation pass with enhanced metrics calculation.
        """
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for sample in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}/{max_epochs-1}"):
                inputs = sample['image'].to(self.device)
                labels = sample['label'].float().to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                
                val_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(outputs.cpu().numpy().flatten())
                
        val_loss = val_loss / len(self.val_loader.dataset)
        metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))
        
        wandb.log({
            "Val Loss": val_loss,
            "MAE": metrics['mae'],
            "RMSE": metrics['rmse'],
            "R2 Score": metrics['r2'],
            "C-Index": metrics['c_index']
        })
        
        print(f"Epoch {epoch}/{max_epochs-1}")
        print(f"Val Loss: {val_loss:.4f}, MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
        print(f"C-Index: {metrics['c_index']:.4f}")
        
        return val_loss, metrics
        
    ## save checkpoints     
    def save_checkpoint(self, epoch, loss, metrics):
        config = self.get_config()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics
        }
        save_path = os.path.join(
            config['logger']['save_dir'],
            config['logger']['save_name'].format(epoch=epoch, loss=loss, metric=metrics['mae'])
        )
        torch.save(checkpoint, save_path)

if __name__ == "__main__":
    trainer = StrokeTimeTrainer()
    trainer.train()