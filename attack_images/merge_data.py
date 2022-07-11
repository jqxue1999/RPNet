from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import torch
from tqdm import tqdm

train_data = CIFAR10("../data", True, transforms.ToTensor())
test_data = CIFAR10("../data", False, transforms.ToTensor())
train_images = torch.load("./train.pth")
test_images = torch.load("./test.pth")

for img, label in tqdm(train_data):
    train_images.append({'tensor': img, 'label': label})

for img, label in tqdm(test_data):
    test_images.append({'tensor': img, 'label': label})

torch.save(train_images, "train_data.pth")
torch.save(test_images, "test_data.pth")
print("end")
