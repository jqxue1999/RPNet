import torch
from models.CIFAR10 import eBaseNet
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
# mean and std for different datasets
IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_transformsFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])

INCEPTION_SIZE = 299
INCEPTION_transformsFORM = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor()])

CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_transformsFORM = transforms.Compose([
    transforms.ToTensor()])

MNIST_SIZE = 28
MNIST_MEAN = [0.1307]
MNIST_STD = [0.3081]
MNIST_transformsFORM = transforms.Compose([
    transforms.ToTensor()])

# add gaussian noise
def add_gaussian_noise(img, sigma=0):
    noise_img = img + sigma * torch.randn_like(img)
    return noise_img


def apply_normalization(imgs, dataset, sigma=0):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        imgs_tensor = add_gaussian_noise(imgs_tensor, sigma)
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor


model = eBaseNet().cuda()
model = torch.nn.DataParallel(model)
checkpoint = torch.load("../checkpoint/CIFAR10/base/eBaseNet.pth")
model.load_state_dict(checkpoint['net'])

dataset = CIFAR10(root="../data", train=False, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
# for batch, (X, y) in enumerate(dataloader):
#     print("end")
#
# print("end")

X0 = torch.load("X0.pth")
X1 = torch.load("X1.pth")
model.eval()
with torch.no_grad():
    model(X0)
    model(X1)