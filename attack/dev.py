import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision.datasets as dset
import utils
import random
import models
from simba_cifar_dev import SimBA
import pandas as pd

base_params = {
    "data_root": "../data",
    "freq_dims": 32,
    "stride": 7,
    "linf_bound": 0.0,
    "order": "rand",
    "num_iters": 0,
    "pixel_attack": False,
    "log_every": 10,
    "seed": 47,
    "num_runs": 1000,
    "batch_size": 128,
    "sigma": 0.0,
    "sigma2": 0.0
}


def defense(params):
    # load model and dataset
    if params["compress"]:
        model = torch.load(params["model_ckpt"])
    else:
        model = getattr(getattr(models, "CIFAR10"), params["model"])().cuda()
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(params["model_ckpt"])
        model.load_state_dict(checkpoint['net'])
    utils.setup_seed(params["seed"])
    if params["output_noise"]:
        model = models.GaussianNoiseNet(model, params["sigma2"])
    model.eval()
    image_size = 32
    testset = dset.CIFAR10(root=params["data_root"], train=False, download=True, transform=utils.CIFAR_TRANSFORM)
    attacker = SimBA(model, 'cifar', image_size, params["sigma"])

    images = torch.zeros(params["num_runs"], 3, image_size, image_size)
    labels = torch.zeros(params["num_runs"]).long()
    preds = labels + 1
    while preds.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        for i in list(idx):
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        preds[idx], _ = utils.get_preds(model, images[idx], 'cifar', batch_size=params["batch_size"], sigma=params["sigma"])

    if params["order"] == 'rand':
        n_dims = 3 * params["freq_dims"] * params["freq_dims"]
    else:
        n_dims = 3 * image_size * image_size
    if params["num_iters"] > 0:
        max_iters = int(min(n_dims, params["num_iters"]))
    else:
        max_iters = int(n_dims)

    i = 0
    upper = min((i + 1) * params["batch_size"], params["num_runs"])
    images_batch = images[(i * params["batch_size"]):upper]
    labels_batch = labels[(i * params["batch_size"]):upper]
    # replace true label with random target labels in case of targeted attack
    if params["targeted"]:
        labels_targeted = labels_batch.clone()
        while labels_targeted.eq(labels_batch).sum() > 0:
            labels_targeted = torch.floor(10 * torch.rand(labels_batch.size())).long()
        labels_batch = labels_targeted
    adv, probs, succs, queries, l2_norms, linf_norms, res = attacker.simba_batch(
        images_batch, labels_batch, max_iters, params["freq_dims"], params["stride"], params["epsilon"],
        linf_bound=params["linf_bound"],
        order=params["order"], targeted=params["targeted"], pixel_attack=params["pixel_attack"],
        log_every=params["log_every"],
        seed=params["seed"])

    return res


if __name__ == "__main__":
    if not os.path.exists('./results/targeted'):
        os.mkdir('./results/targeted')
    if not os.path.exists('./results/untargeted'):
        os.mkdir('./results/untargeted')
    dev_params = [
        # {
        #     "targeted": True,
        #     "compress": False,
        #     "output_noise": False,
        #     "model_ckpt": "../checkpoint/CIFAR10/BaseNet.pth",
        #     "model": "BaseNet"
        # },
        {
            "targeted": True,
            "compress": True,
            "output_noise": False,
            "model_ckpt": "../checkpoint/CIFAR10/eBaseNet-10.pth",
            "model": "eBaseNet"
        },
        # {
        #     "targeted": False,
        #     "compress": False,
        #     "output_noise": False,
        #     "model_ckpt": "../checkpoint/CIFAR10/BaseNet.pth",
        #     "model": "BaseNet"
        # },
        {
            "targeted": False,
            "compress": True,
            "output_noise": False,
            "model_ckpt": "../checkpoint/CIFAR10/eBaseNet-10.pth",
            "model": "eBaseNet"
        }
    ]
    epsilons = [0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0]
    for params in dev_params:
        df = pd.DataFrame({100: None, 200: None, 300: None, 400: None, 500: None, 1000: None}, index=epsilons)
        for epsilon in epsilons:
            params["epsilon"] = epsilon
            params.update(base_params)
            a = defense(params)
            df.loc[epsilon] = list(a.values())
        df.to_csv("./results/{}/{}.csv".format("targeted" if params["targeted"] else "untargeted", params["model"]))
        print("{}, {} saved".format("targeted" if params["targeted"] else "untargeted", params["model"]))