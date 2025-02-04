import torch.nn as nn
from monai.networks.nets import resnet101, resnet50, resnet18, ViT
import torch


class Classifier(nn.Module):
    """ Classifier class with FC layer and single output neuron """
    def __init__(self, d_model, hidden_dim=1024, num_classes=1):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.fc(x)
        return x


class Backbone(nn.Module):
    """ ResNet 3D Backbone"""

    def __init__(self):
        super(Backbone, self).__init__()

        resnet = resnet50(pretrained=False)  # assuming you're not using a pretrained model
        resnet.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        hidden_dim = resnet.fc.in_features
        self.backbone = resnet
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x


class SingleScanModel(nn.Module):
    """ End to end model with backbone and classifier"""

    def __init__(self, backbone, classifier):
        super(SingleScanModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)

        
    def forward(self, x):
       
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class SingleScanModelBP(nn.Module):
    """ End to end model with backbone and classifier that takes 2 input scans at once"""

    def __init__(self, backbone, classifier):
        super(SingleScanModelBP, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)
        self.bilinear_pooling = nn.Bilinear(in1_features=2048, in2_features=2048, out_features=512)

        
    def forward(self, x):
        x = [self.backbone(scan) for scan in x.split(1, dim=1)]
        features = torch.stack(x, dim=1).squeeze(2)
        merged_features = torch.mean(features, dim=1)  # Shape: (batch_size, feature_dim)
        merged_features = self.dropout(merged_features)
        output = self.classifier(merged_features)
        return output