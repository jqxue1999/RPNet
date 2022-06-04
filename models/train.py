import argparse
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import models
from models.utils import setup_seed, train_loop, test_loop, get_dataset

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset name')
parser.add_argument('--dataset_dir', default='./data', type=str, help='dataset dir')
parser.add_argument('--model', default="BaseNet", type=str, help='mdoel name')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--save_dir', required=True, type=str, help='save location')
args = parser.parse_args()

setup_seed(47)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
train_dataloader, test_dataloader = get_dataset(args.dataset, args.dataset_dir, args.batch_size)

model = getattr(getattr(models, args.dataset), args.model)().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(args.epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_acc, test_loss = test_loop(test_dataloader, model, loss_fn, device)
    if test_acc > best_acc:
        state = {
            'net': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        torch.save(state, args.save_dir)
        best_acc = test_acc
    print("Test: \n Best Acc:{:.1f}%, Acc: {:.1f}%, Avg loss: {:.8f}".format(100 * best_acc, 100 * test_acc, test_loss))

    scheduler.step()
print("Done!")
