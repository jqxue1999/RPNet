import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def get_dataset(dataset_name, dataset_dir, batch_size=64):
    if dataset_name == "CIFAR10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_data = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform_test)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    elif dataset_name == "MNIST":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_data = torchvision.datasets.MNIST(root=dataset_dir, train=False, download=True, transform=transform_test)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    elif dataset_name == "SkinCancer":
        transforms_test = transforms.ToTensor()

        class TestData(torch.utils.data.Dataset):
            def __init__(self, df_dir, transform=None):
                super().__init__()
                self.data_array = pd.read_csv(df_dir).to_numpy()
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

        test_data = TestData(os.path.join(dataset_dir, 'SkinCancer', 'test.csv'), transforms_test)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    elif dataset_name == "DiabeticRetinopathy":
        transforms_test = transforms.ToTensor()

        class TestData(torch.utils.data.Dataset):
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

        test_data = TestData(dataset_dir, transforms_test)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    else:
        test_dataloader = None

    return test_dataloader


def test(model, dataloader):
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