import torch
from torch import nn
from models.Square import Square


class eBaseNet(nn.Module):
    def __init__(self):
        super(eBaseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 5, 5, 2, 2), Square()
        )

        self.classifier = nn.Sequential(
            nn.Linear(980, 100), Square(),
            nn.Linear(100, 10)
        )

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 5, 5, 2, 2), nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(980, 100), nn.ReLU(),
            nn.Linear(100, 10)
        )

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    X = torch.ones((64, 1, 32, 32))
    model = BaseNet()
    y = model(X)
    print(y.shape)
