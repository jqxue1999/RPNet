import torch
from compressed.utils import get_dataset, test
from models.CIFAR10 import BaseNet, eBaseNet
from models.GaussianNoise import GaussianNoiseNet

if __name__ == '__main__':
    testset = get_dataset("CIFAR10", "../data")
    model = torch.load("../checkpoint/CIFAR10/eBaseNet-10.pth")
    # model = BaseNet()
    # model = torch.nn.DataParallel(model)
    # checkpoint = torch.load("../checkpoint/CIFAR10/BaseNet.pth")
    # model.load_state_dict(checkpoint['net'])
    sigma = 0.25
    if sigma != 0:
        model = GaussianNoiseNet(model, sigma)
    test(model, testset)