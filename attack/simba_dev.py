import argparse
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torchvision import datasets
import utils
import random
import models
from simba import SimBA


def defense(params):
    # load model and dataset
    model = torch.load(params["model_ckpt"])
    utils.setup_seed(params["seed"])
    model = models.GaussianNoiseNet(model, params["sigma2"])
    if params["dataset"] == "mnist":
        testset = datasets.MNIST(root=params["data_root"], train=False, download=True, transform=utils.MNIST_TRANSFORM)
    elif params["dataset"] == "SkinCancer":
        testset = utils.SkinCancerData(os.path.join(params["data_root"], 'SkinCancer', 'test.csv'),
                                       utils.SkinCancer_TRANSFORM)
    elif params["dataset"] == "cifar":
        testset = datasets.CIFAR10(root=params["data_root"], train=False, download=True,
                                   transform=utils.CIFAR_TRANSFORM)
    elif params["dataset"] == "DiabeticRetinopathy":
        testset = utils.DiabeticRetinopathyData(params["data_root"], utils.DiabeticRetinopathy_TRANSFORM)
    channel = 1 if params["dataset"] == "mnist" else 3
    attacker = SimBA(model, params["dataset"], params["image_size"], params["sigma"])
    images = torch.zeros(params["num_runs"], channel, params["image_size"], params["image_size"])
    labels = torch.zeros(params["num_runs"]).long()
    preds = labels + 1
    while preds.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        for i in list(idx):
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        preds[idx], _ = utils.get_preds(model, images[idx], params["dataset"], batch_size=params["batch_size"],
                                        sigma=params["sigma"])

    n_dims = channel * params["image_size"] * params["image_size"]
    if params["num_iters"] > 0:
        max_iters = int(min(n_dims, params["num_iters"]))
    else:
        max_iters = int(n_dims)

    i = 0
    upper = min((i + 1) * params["batch_size"], params["num_runs"])
    images_batch = images[(i * params["batch_size"]):upper]
    labels_batch = labels[(i * params["batch_size"]):upper]
    if params["dataset"] == "mnist":
        labels_set = list(range(10))
    elif params["dataset"] == "cifar":
        labels_set = list(range(10))
    elif params["dataset"] == "DiabeticRetinopathy":
        labels_set = list(range(5))
    elif params["dataset"] == "SkinCancer":
        labels_set = list(range(7))
    # replace true label with random target labels in case of targeted attack
    if params["targeted"]:
        labels_batch = utils.get_target_labels(labels_batch, labels_set)
    res, l2_norms = attacker.simba_batch(images_batch, labels_batch, max_iters, params["epsilon"],
                                         linf_bound=params["linf_bound"], targeted=params["targeted"],
                                         seed=params["seed"], T=params["T"], beta_range=params["beta_range"])

    return res

parser = argparse.ArgumentParser(description='Runs PNet-Attack on a set of images')
parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')
parser.add_argument('--seed', type=int, default=47, help='random seed')
parser.add_argument('--num_runs', type=int, default=128, help='number of image samples')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for parallel runs')
parser.add_argument('--data_root', type=str, default='./data', help='root directory of imagenet data')
parser.add_argument('--dataset', type=str, default='cifar', help='root directory of imagenet data')
parser.add_argument('--model_type', type=str, default='CIFAR10', help='root directory of imagenet data')
parser.add_argument('--image_size', type=int, default=32, help='the size of image, 32 for cifar10')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
parser.add_argument('--model_ckpt', type=str, required=True, help='model checkpoint location')
parser.add_argument('--sigma1', type=float, default=0.0, help='gaussian noise on input layer')
parser.add_argument('--sigma2', type=float, default=0.0, help='gaussian noise on output layer')
parser.add_argument('--epsilon', type=float, default=1.0, help='attack epsilon')
parser.add_argument('--T', type=float, default=300, help='the cycle of schedule')
parser.add_argument('--beta_max', type=float, default=1.5, help='beta_max')
parser.add_argument('--beta_min', type=float, default=0.5, help='beta_min')
args = parser.parse_args()


dev_params = {
    "linf_bound": args.linf_bound,
    "num_iters": args.num_iters,
    "seed": args.seed,
    "num_runs": args.num_runs,
    "batch_size": args.batch_size,
    "data_root": args.data_root,
    "dataset": args.dataset,
    "model_type": args.model_type,
    "image_size": args.image_size,
    "targeted": args.targeted,
    "model_ckpt": args.model_ckpt,
    "sigma": args.sigma1,
    "sigma2": args.sigma2,
    "epsilon": args.epsilon,
    "T": args.T,
    "beta_range": [args.beta_min, args.beta_max]
}
a = defense(dev_params)
