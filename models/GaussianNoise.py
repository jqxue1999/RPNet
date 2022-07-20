import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        y = x + self.sigma * torch.randn_like(x)
        return y
    # def forward(self, x, max, min):
    #     scale = (max-min).reshape(-1, 1).repeat(1, 10)
    #     y = x + self.sigma * scale * torch.randn_like(x)
    #     return y


class GaussianNoiseNet(nn.Module):
    def __init__(self, base, sigma):
        super(GaussianNoiseNet, self).__init__()
        self.base = base
        self.softmax = nn.Softmax(dim=1)
        self.GaussianNoise = GaussianNoise(sigma)

    def forward(self, x):
        x = self.base(x)
        x = self.softmax(x)
        y = self.GaussianNoise(x)
        return y

    # def forward(self, x):
    #     x = self.base(x)
    #     y = self.GaussianNoise(x, x.max(dim=1)[0], x.min(dim=1)[0])
    #     return y