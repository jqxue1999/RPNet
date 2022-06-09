import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms


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