from torchvision import models
import torch.nn as nn

def resnet18():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2),
            nn.LogSoftmax(dim=1)
        )
    return model
