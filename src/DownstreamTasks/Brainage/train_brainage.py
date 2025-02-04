import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm 
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_absolute_error
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset2 import MedicalImageDatasetBalancedIntensity3D, TransformationMedicalImageDatasetBalancedIntensity3D
from model import Backbone, SingleScanModel, Classifier
from utils import BaseConfig


class BrainAgeTrainer(BaseConfig):
    """
    A trainer class for brain age prediction models.
    
    This class handles the complete training pipeline including model setup,
    data loading, training loop, and validation. 
    Inherits from BaseConfig for configuration management.
    """

    def __init__(self):
        """Initialize the trainer with model, data, and training setup."""
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
        
    def setup_model(self):
        """
        Set up the model architecture.
        
        Initializes the backbone and classifier blocks, and loads
        checkpoints
        """
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
        Set up data loaders for training and validation.
        Inherit configuration from the base config
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
        Set up training config with loss, scheduler, optimizer.
        """
        config = self.get_config()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['optim']['lr'],
            weight_decay=config["optim"]["weight_decay"]
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
        self.scaler = GradScaler()
        
    def train(self):
        """
        main training loop
        """
        config = self.get_config()
        max_epochs = config['optim']['max_epochs']
        best_val_loss = float('inf')
        best_val_mae = float('inf')

        for epoch in range(max_epochs):
            train_loss = self.train_epoch(epoch, max_epochs)
            val_loss, mae = self.validate_epoch(epoch, max_epochs)
            
            # Save best model
            if (val_loss <= best_val_loss) and (mae <= best_val_mae):
                print(f"Improved Val Loss from {best_val_loss:.4f} to {val_loss:.4f}")
                print(f"Improved Val MAE from {best_val_mae:.4f} to {mae:.4f}")
                best_val_loss = val_loss
                best_val_mae = mae
                self.save_checkpoint(epoch, val_loss, mae)
                
        wandb.finish()
                
    def train_epoch(self, epoch, max_epochs):
        """
        Train pass.
        
        Args:
            epoch (int): Current epoch number
            max_epochs (int): Total number of epochs
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        train_loss = 0.0
        
        for sample in tqdm(self.train_loader, desc=f"Training Epoch {epoch}/{max_epochs-1}"):
            inputs = sample['image'].to(self.device)
            labels = sample['label'].float().to(self.device)
            
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(self.train_loader.dataset)
        wandb.log({"Train Loss": train_loss})
        return train_loss
        
    def validate_epoch(self, epoch, max_epochs):
        """
        Validation pass.
        
        Args:
            epoch (int): Current epoch number
            max_epochs (int): Total number of epochs
            
        Returns:
            tuple: (validation_loss, mean_absolute_error)
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
        mae = mean_absolute_error(all_labels, all_preds)
        
        wandb.log({"Val Loss": val_loss, "MAE": mae})
        self.scheduler.step(val_loss)
        
        print(f"Epoch {epoch}/{max_epochs-1} Val Loss: {val_loss:.4f} MAE: {mae:.4f}")
        return val_loss, mae
        
    def save_checkpoint(self, epoch, loss, mae):
        """
        Save model checkpoint.
        """
        config = self.get_config()
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'epoch': epoch,
        }
        save_path = os.path.join(
            config['logger']['save_dir'],
            config['logger']['save_name'].format(epoch=epoch, loss=loss, metric=mae)
        )
        torch.save(checkpoint, save_path)

if __name__ == "__main__":
    trainer = BrainAgeTrainer()
    trainer.train()