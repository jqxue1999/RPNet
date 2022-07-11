import torch
import torchvision
from torchvision.transforms import transforms
import numpy as np
import random


def add_gaussian_noise(img, sigmas, is_single=False):
    if is_single:
        sigma = sigmas[0] * random.random()
    else:
        sigma = random.choice(sigmas)
    noise_img = img + sigma * torch.randn_like(img)
    return noise_img


def get_dataset(dataset_name, dataset_dir, batch_size=64, sigmas=[0], is_single=False):
    if dataset_name == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: add_gaussian_noise(img, sigmas, is_single)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_data = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    elif dataset_name == "MNIST":
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081,))
        ])
        train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    else:
        train_dataloader, test_dataloader = None, None

    return train_dataloader, test_dataloader


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return correct, test_loss


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
