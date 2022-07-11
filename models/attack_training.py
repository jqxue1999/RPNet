import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
import models
from models.utils import setup_seed, train_loop, test_loop

# env
setup_seed(47)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
epochs = 500
lr = 0.001
save_dir = "../checkpoint/CIFAR10/attack_training/eBaseNet.pth"


# dataset, dataloader
class AttackData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.images_labels = torch.load(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images_labels)

    def __getitem__(self, index):
        image = self.images_labels[index].get("tensor")
        label = int(self.images_labels[index].get("label"))
        if self.transform:
            image = self.transform(image)
        return image, label


train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_transforms = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_data = AttackData("../attack_images/train_data.pth", train_transforms)
test_data = AttackData("../attack_images/test_data.pth", test_transforms)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# model
model = getattr(getattr(models, "CIFAR10"), "eBaseNet")().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# train and test
best_acc = 0
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_acc, test_loss = test_loop(test_dataloader, model, loss_fn, device)
    if test_acc > best_acc:
        state = {
            'net': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        torch.save(state, save_dir)
        best_acc = test_acc
    print("Test: \n Best Acc:{:.1f}%, Acc: {:.1f}%, Avg loss: {:.8f}".format(100 * best_acc, 100 * test_acc, test_loss))

    scheduler.step()
print("Done!")
