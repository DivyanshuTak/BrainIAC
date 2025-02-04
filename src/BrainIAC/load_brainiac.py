import torch
from model import ResNet50_3D
import argparse

def load_brainiac(checkpoint_path, device='cuda'):
    """
    Load the ResNet50 model and BrainIAC checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model with checkpoint weights
    """
    # spinup the model 
    model = ResNet50_3D()
    
    # Load brainiac weights 
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]
    filtered_state_dict = {key: value for key, value in state_dict.items() if 'backbone' in key}
    model.load_state_dict(filtered_state_dict)
    print("BrainIAC Loaded!!")
    
    return model

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description='Load backbone model from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to load the model on (cuda or cpu)')
    args = parser.parse_args()
    
    # Load model
    model = load_brainiac(args.checkpoint, args.device)
    print(f"Model loaded successfully from {args.checkpoint}!")