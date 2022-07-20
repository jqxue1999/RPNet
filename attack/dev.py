import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision.datasets as dset
import utils
import random
import models
from simba_cifar_dev import SimBA
import numpy as np
import pandas as pd
from PIL import Image

base_params = {
    "model": "eBaseNet",
    "compress": True,
    "linf_bound": 0.0,
    "order": "rand",
    "num_iters": 0,
    "pixel_attack": False,
    "log_every": 10,
    "seed": 47,
    "num_runs": 1000,
    "batch_size": 128
}


class SkinCancerData(torch.utils.data.Dataset):
    def __init__(self, df_dir, transform=None):
        super().__init__()
        self.data_array = pd.read_csv(df_dir).to_numpy()
        self.transform = transform

    def __len__(self):
        return self.data_array.shape[0]

    def __getitem__(self, index):
        # Load data and get label
        X = Image.fromarray(np.uint8(self.data_array[index, :-1].reshape(28, 28, 3)))
        y = torch.tensor(int(self.data_array[index, -1]))

        if self.transform:
            X = self.transform(X)

        return X, y


class DiabeticRetinopathyData(torch.utils.data.Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = os.path.join(base_dir, 'images')
        self.df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(os.path.join(self.base_dir, self.df['id_code'][index] + ".png"))
        y = torch.tensor(int(self.df['diagnosis'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


def defense(params):
    # load model and dataset
    if params["compress"]:
        model = torch.load(params["model_ckpt"])
    else:
        model = getattr(getattr(models, params["model_type"]), params["model"])().cuda()
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(params["model_ckpt"])
        model.load_state_dict(checkpoint['net'])
    utils.setup_seed(params["seed"])
    if params["output_noise"]:
        model = models.GaussianNoiseNet(model, params["sigma2"])
    model.eval()
    if params["dataset"] == "SkinCancer":
        testset = SkinCancerData(os.path.join(params["data_root"], 'SkinCancer', 'test.csv'),
                                 utils.SkinCancer_TRANSFORM)
    elif params["dataset"] == "cifar":
        testset = dset.CIFAR10(root=params["data_root"], train=False, download=True, transform=utils.CIFAR_TRANSFORM)
    elif params["dataset"] == "DiabeticRetinopathy":
        testset = DiabeticRetinopathyData(params["data_root"], utils.DiabeticRetinopathy_TRANSFORM)
    attacker = SimBA(model, params["dataset"], params["image_size"], params["sigma"])

    images = torch.zeros(params["num_runs"], 3, params["image_size"], params["image_size"])
    labels = torch.zeros(params["num_runs"]).long()
    preds = labels + 1
    while preds.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        for i in list(idx):
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        preds[idx], _ = utils.get_preds(model, images[idx], params["dataset"], batch_size=params["batch_size"],
                                        sigma=params["sigma"])

    if params["order"] == 'rand':
        n_dims = 3 * params["freq_dims"] * params["freq_dims"]
    else:
        n_dims = 3 * params["image_size"] * params["image_size"]
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
    if not os.path.exists('./results'):
        os.mkdir('./results')

    dev_params = [
        {
            "data_root": "../data",
            "dataset": "SkinCancer",
            "model_type": "SkinCancer",
            "freq_dims": 28,
            "image_size": 28,
            "stride": 7,
            "version": "sigma_single_output=0.06",
            "targeted": True,
            "output_noise": False,
            "model_ckpt": "../checkpoint/SkinCancer/sigma_single/eBaseNet-16.pth",
            "sigma": 0.05,
            "sigma2": 0.06,
            "epsilon": 1.0
        },
        {
            "data_root": "../data",
            "dataset": "SkinCancer",
            "model_type": "SkinCancer",
            "freq_dims": 28,
            "image_size": 28,
            "stride": 7,
            "version": "sigma_single_output=0.06",
            "targeted": False,
            "output_noise": True,
            "model_ckpt": "../checkpoint/SkinCancer/sigma_single/eBaseNet-16.pth",
            "sigma": 0.05,
            "sigma2": 0.06,
            "epsilon": 1.0
        }
    ]
    # sigmas = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    # for params in dev_params:
    #     df = pd.DataFrame({100: None, 200: None, 300: None, 400: None, 500: None, 1000: None}, index=sigmas)
    #     for sigma in sigmas:
    #         base_params["sigma"] = sigma
    #         params.update(base_params)
    #         a = defense(params)
    #         df.loc[sigma] = list(a.values())
    #     df.to_csv("./results/{}/epsilon/epsilon={}_{}_{}.csv".format(params["model_type"], params["epsilon"],
    #         params["version"], "targeted" if params["targeted"] else "untargeted"))
    #     print("./results/{}/epsilon/epsilon={}_{}_{}.csv".format(params["model_type"], params["epsilon"],
    #         params["version"], "targeted" if params["targeted"] else "untargeted"))

    # sigmas2 = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    # for params in dev_params:
    #     df = pd.DataFrame({100: None, 200: None, 300: None, 400: None, 500: None, 1000: None}, index=sigmas2)
    #     for sigma2 in sigmas2:
    #         base_params["sigma2"] = sigma2
    #         params.update(base_params)
    #         a = defense(params)
    #         df.loc[sigma2] = list(a.values())
    #     save_dir = "./results/{}/epsilon/epsilon={}_{}_{}_output.csv".format(params["model_type"], params["epsilon"],
    #                                                                          params["version"], "targeted" if params[
    #                                                                          "targeted"] else "untargeted")
    #     df.to_csv(save_dir)
    #     print(save_dir)

    epsilons = [0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0]
    for params in dev_params:
        df = pd.DataFrame({100: None, 200: None, 300: None, 400: None, 500: None, 1000: None}, index=epsilons)
        for epsilon in epsilons:
            base_params["epsilon"] = epsilon
            params.update(base_params)
            a = defense(params)
            df.loc[epsilon] = list(a.values())
        df.to_csv("./results/{}/sigma/sigma={}_{}_{}.csv".format(params["model_type"], params["sigma"], params["version"], "targeted" if params["targeted"] else "untargeted"))
        print("./results/{}/sigma/sigma={}_{}_{}.csv".format(params["model_type"], params["sigma"], params["version"], "targeted" if params["targeted"] else "untargeted"))