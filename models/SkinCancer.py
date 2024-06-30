import torch
from torch import nn
from models.Square import Square


class eBaseNet(nn.Module):
    def __init__(self):
        super(eBaseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), Square(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3), Square(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3), Square()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128), Square(), nn.Dropout(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x