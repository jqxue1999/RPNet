import argparse
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
from models import GaussianNoiseNet
import torchvision.transforms as transforms
import torchvision
from utils import add_gaussian_noise


class DTestData(torch.utils.data.Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = os.path.join(base_dir, 'images')
        self.df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(os.path.join(self.base_dir, self.df['id_code'][index] + ".png"))
        y = torch.tensor(int(self.df['diagnosis'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


class TestData(torch.utils.data.Dataset):
    def __init__(self, df_dir, transform=None):
        super().__init__()
        self.data_array = pd.read_csv(os.path.join(df_dir, 'test.csv')).to_numpy()
        self.transform = transform

    def __len__(self):
        return self.data_array.shape[0]

    def __getitem__(self, index):
        # Load data and get label
        X = Image.fromarray(np.uint8(self.data_array[index, :-1].reshape(28, 28, 3)))
        y = torch.tensor(int(self.data_array[index, -1]))

        if self.transform:
            X = self.transform(X)

        return X, y


def add_gaussian_noise(img, sigma):
    noise_img = img + sigma * torch.randn_like(img)
    return noise_img


def get_dataset(data_name, data_dir, batch_size=64, sigma=0):
    if data_name == "CIFAR10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img: add_gaussian_noise(img, sigma)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    elif data_name == "SkinCancer":
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img: add_gaussian_noise(img, sigma))
        ])

        test_data = TestData(data_dir, transforms_test)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    elif data_name == "DiabeticRetinopathy":
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.Lambda(lambda img: add_gaussian_noise(img, sigma))
        ])

        test_data = DTestData(data_dir, transforms_test)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return test_dataloader


def test(model, dataloader, sigma2=0):
    if sigma2 != 0:
        model = GaussianNoiseNet(model, sigma2)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model = model.cuda()
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print("Acc: {0}, Loss: {1}".format(correct, test_loss))
    return correct, test_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_dir', type=str)

    args = parser.parse_args()

    model = torch.load(args.model_dir)
    sigma = 0.009
    for sigma2 in [0.0, 0.003, 0.009, 0.03, 0.09]:
        testset = get_dataset(args.data_name, args.data_dir, 128, sigma)
        print("sigma = {}".format(sigma2))
        test(model, testset, sigma2)
