from torch import nn
from models.Square import Square
import torch


class eBaseNet(nn.Module):
    def __init__(self):
        super(eBaseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), Square(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), Square(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), Square()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1600, 128), Square(), nn.Dropout(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1600, 128), nn.ReLU(), nn.Dropout(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    X = torch.ones((64, 3, 32, 32))
    model = BaseNet()
    y = model(X)
    print(y.shape)
