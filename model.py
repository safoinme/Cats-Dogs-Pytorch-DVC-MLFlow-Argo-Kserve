import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock


class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2,2,2,2], num_classes=2)
        self.fc = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def pretrained(self):
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )
        return model