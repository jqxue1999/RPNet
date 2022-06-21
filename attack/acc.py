import torch
from models.CIFAR10 import BaseNet, eBaseNet
import torchvision.transforms as transforms
import torchvision
from utils import add_gaussian_noise
import random


def add_gaussian_noise(img, sigmas):
    sigma = random.choice(sigmas)
    noise_img = img + sigma * torch.randn_like(img)
    return noise_img


def get_dataset(dataset_dir, batch_size=64, sigmas=[0]):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: add_gaussian_noise(img, sigmas)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_data = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform_test)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

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


if __name__ == '__main__':
    # model = torch.load("../checkpoint/CIFAR10/sigmas/eBaseNet-10.pth")
    model = BaseNet()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load("../checkpoint/CIFAR10/sigmas/BaseNet.pth")
    model.load_state_dict(checkpoint['net'])
    for sigma in [0.0, 0.003, 0.009, 0.03, 0.09, 0.13, 0.19, 0.33, 0.39, 0.63, 0.69]:
        testset = get_dataset("../data", 128, [sigma])
        # if sigma != 0:
        #     model = GaussianNoiseNet(model, sigma)
        print("sigma = {}".format(sigma))
        test(model, testset)
