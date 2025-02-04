import torch.nn as nn
#from utils.get_feature_extactor import unetr_feature_extractor, Simclr_feature_extractor
from monai.networks.nets import resnet101, resnet50, resnet18, ViT



## resnet50 architecture, FC layers converted to I
class ResNet50_3D(nn.Module):
    def __init__(self):
        super(ResNet50_3D, self).__init__()

        resnet = resnet50(pretrained=False)  # assuming you're not using a pretrained model
        resnet.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        hidden_dim = resnet.fc.in_features
        self.backbone = resnet
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x

