from compressed import utils
import models.CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms
from compressed import compress
import matplotlib.pyplot as plt
import os


def get_acc_loss(model, model_dir, data_dir):
    Acc, Loss = [], []
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)

    utils.ensure_dir(model_dir)
    model_raw = getattr(models.CIFAR10, model)().cuda()
    model_raw = torch.nn.DataParallel(model_raw)
    checkpoint = torch.load(model_dir)
    model_raw.load_state_dict(checkpoint['net'])
    acc, loss = utils.test(model_raw, test_dataloader)
    Acc.append(acc), Loss.append(loss)

    for i in range(1, 16):
        model_raw.load_state_dict(checkpoint['net'])
        model_new = compress.CompressedModel(model_raw, input_scale=1, act_bits=i, weight_bits=i)
        utils.test(model_new, test_dataloader)
        model_new.quantize_params()
        acc, loss = utils.test(model_new, test_dataloader)
        Acc.append(acc), Loss.append(loss)

    return Acc, Loss


def plot_score(model, img_dir):
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    X = list(range(1, 16))
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.add_subplot(111)
    lin1 = ax.plot(X, model.get("Acc")[1:16], label="Acc", color="blue", linewidth=3)
    hlin1 = ax.plot(X, [model.get("Acc")[0]] * 15, color="blue", label="Original Acc", linewidth=3, linestyle=':')
    ax.set_title("{}: Acc {}, Loss {}".format(model.get("name"), model.get("Acc")[0], model.get("Loss")[0]), size=20)
    ax.set_xlabel("bits", size=18)
    ax.set_ylabel("Acc", size=18)

    ax1 = ax.twinx()
    lin2 = ax1.plot(X, model.get("Loss")[1:16], color="red", label="Loss", linewidth=3)
    hlin2 = ax1.plot(X, [model.get("Loss")[0]] * 15, color="red", label="Original Loss", linewidth=3, linestyle=':')
    ax1.set_ylabel("Loss", size=18)

    lins = lin1 + lin2 + hlin1 + hlin2
    labs = [l.get_label() for l in lins]
    ax.legend(lins, labs, loc="upper left", fontsize=15)
    plt.savefig("{}{}".format(img_dir, model.get("name")))
    plt.show()


if __name__ == '__main__':
    BaseAcc, BaseLoss = get_acc_loss("BaseNet", "../checkpoint/CIFAR10/BaseNet.pth", "../data")
    eBaseAcc, eBaseLoss = get_acc_loss("eBaseNet", "../checkpoint/CIFAR10/eBaseNet.pth", "../data")
    Base = {"Acc": BaseAcc, "Loss": BaseLoss, "name": "BaseNet"}
    eBase = {"Acc": eBaseAcc, "Loss": eBaseLoss, "name": "eBaseNet"}
    plot_score(Base, "./image/")
    plot_score(eBase, "./image/")
    print("End")
