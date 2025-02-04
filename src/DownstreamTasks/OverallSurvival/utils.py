import yaml
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.set_float32_matmul_precision("medium")
        
    def custom_collate(self, batch):
        """Handles variable size of the scans and pads the sequence dimension."""
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]
        
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

        return {"image": torch.stack(padded_images, dim=0), "label": torch.stack(labels)}

    def get_config(self):
        return self.config
    


def plot_km_curve(data, time="survival_time", event="deadstatus_event", group="group", time_limit=4):
    df = data.copy()
    df.loc[df[time] >= time_limit, event] = 0  # Cap events at time_limit years
    T, E, G = df[time], df[event], df[group]

    kmfs = {}
    for grp in sorted(df[group].unique()):
        kmf = KaplanMeierFitter().fit(T[df[group] == grp], E[df[group] == grp], label=grp)
        kmfs[grp] = kmf

    results = multivariate_logrank_test(T, G, event_observed=E)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    for grp, kmf in kmfs.items():
        kmf.plot(ax=ax, show_censors=True)

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival Probability")
    ax.set_xlim(0, time_limit)
    ax.legend()

    p_val = results.p_value
    ax.text(0.7, 0.9, f"p = {p_val:.3f}", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    return fig

def calculate_metrics(pred_probs, pred_labels, true_labels):
    """
    classification metrics.
    Args:
        pred_probs (numpy.ndarray): Predicted probabilities
        pred_labels (numpy.ndarray): Predicted labels
        true_labels (numpy.ndarray): Ground truth labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1, and AUC metrics
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