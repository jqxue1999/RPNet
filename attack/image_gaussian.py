from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import transforms
import models.utils as utils
import torch
import os


def transform_data(sigma):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: utils.add_gaussian_noise(img, [sigma]))
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform_train


if __name__ == "__main__":
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    writer = SummaryWriter("logs")
    dataset_dir = "../data"
    dataset = {
        '0': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform_data(0)),
        '0.003': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                              transform=transform_data(0.003)),
        '0.009': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                              transform=transform_data(0.009)),
        '0.03': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                             transform=transform_data(0.03)),
        '0.09': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                             transform=transform_data(0.09)),
        '0.13': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                             transform=transform_data(0.13)),
        '0.19': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                             transform=transform_data(0.19)),
        '0.33': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                             transform=transform_data(0.33)),
        '0.39': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                             transform=transform_data(0.39)),
        '0.63': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                             transform=transform_data(0.63)),
        '0.69': torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                             transform=transform_data(0.69))
    }

    for sigma in dataset.keys():
        dataloader = torch.utils.data.DataLoader(dataset.get(sigma), batch_size=64, shuffle=False)
        step = 0
        for data in dataloader:
            imgs, targets = data
            writer.add_images("sigma = {}".format(sigma), imgs, step)
            step += 1
            if step == 1:
                break

    writer.close()
