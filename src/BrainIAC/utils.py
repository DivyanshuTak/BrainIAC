import yaml
import os
import torch
import random
import numpy as np

class BaseConfig:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.setup_environment()
    
    def setup_environment(self):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config["gpu"]["visible_device"]
        self.device =  torch.device(self.config["gpu"]["device"]) #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.set_float32_matmul_precision("medium")
        
    def custom_collate(self, batch):
        """Handles variable size of the scans and pads the sequence dimension."""
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]
        pat_ids = [item['pat_id'] for item in batch]

        max_len = self.config["data"]["collate"]  # Single scan input
        padded_images = []
        
        for img in images:
            pad_size = max_len - img.shape[0]
            if pad_size > 0:
                padding = torch.zeros((pad_size,) + img.shape[1:])
                img_padded = torch.cat([img, padding], dim=0)
                padded_images.append(img_padded)
            else:
                padded_images.append(img)

        return {"image": torch.stack(padded_images, dim=0), "label": torch.stack(labels),
                        "pat_id": pat_ids 
}

    def get_config(self):
        return self.config