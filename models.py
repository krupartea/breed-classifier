import torch.nn as nn
from torchvision.models.resnet import resnet152, ResNet152_Weights
import torch


class Resnet(nn.Module):
    def __init__(
            self,
            num_classes,
    ):
        super().__init__()

        self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # replace ResNet's last fc layer with such that suits `num_classes`
        in_features = list(self.resnet.children())[-1].in_features
        self.classifier = nn.Linear(in_features, num_classes)
        # freeze ResNet's weights
        for param in self.resnet.parameters():
            param.requires_grad = False
        # remove ResNet's last fc layer (we'll use our `self.classifier` instead)
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.resnet(x)
        x = torch.reshape(x, (x.shape[0], -1))  # flatten, preserving batches
        x = self.classifier(x)
        return x


class Toy(nn.Module):
    def __init__(
            self,
            num_classes,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, num_classes, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class BreedClassifier(nn.Module):
    def __init__(
            self,
            num_classes,
    ):
        super().__init__()

        self.conv = nn.Sequential(

            # kinda block
            nn.Conv2d(3, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # one more block
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            # classifier
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x
