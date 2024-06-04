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