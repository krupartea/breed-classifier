import torch.nn as nn


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

            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x