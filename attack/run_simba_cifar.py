import sys
import os
import pandas as pd
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision.datasets as dset
import utils
import math
import random
import argparse
import models
from simba_cifar import SimBA
from tqdm import tqdm

class DiabeticRetinopathyData(torch.utils.data.Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = os.path.join(base_dir, 'images')
        self.df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
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

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
parser.add_argument('--data_root', type=str, default='./data', help='root directory of imagenet data')
parser.add_argument('--result_dir', type=str, default='save_cifar', help='directory for saving results')
parser.add_argument('--sampled_image_dir', type=str, default='save_cifar', help='directory to cache sampled images')
parser.add_argument('--model', type=str, required=True, help='type of base model to use')
parser.add_argument('--compress', action='store_true', help='compress model or normal model')
parser.add_argument('--model_ckpt', type=str, required=True, help='model checkpoint location')
parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for parallel runs')
parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')
parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
parser.add_argument('--sigma', type=float, default=0.0, help='gaussian noise')
parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
parser.add_argument('--freq_dims', type=int, default=32, help='dimensionality of 2D frequency space')
parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
parser.add_argument('--stride', type=int, default=7, help='stride for block order')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
parser.add_argument('--seed', type=int, default=47, help='type of base model to use')
args = parser.parse_args()

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if not os.path.exists(args.sampled_image_dir):
    os.mkdir(args.sampled_image_dir)

# load model and dataset
if args.compress:
    model = torch.load(args.model_ckpt)
else:
    model = getattr(getattr(models, "DiabeticRetinopathy"), args.model)().cuda()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.model_ckpt)
    model.load_state_dict(checkpoint['net'])

utils.setup_seed(args.seed)
model.eval()
image_size = 32
testset = DiabeticRetinopathyData(args.data_root, utils.DiabeticRetinopathy_TRANSFORM)
attacker = SimBA(model, 'DiabeticRetinopathy', image_size, args.sigma)

# load sampled images or sample new ones
# this is to ensure all attacks are run on the same set of correctly classified images
batchfile = '%s/images_%s_%d.pth' % (args.sampled_image_dir, args.model, args.num_runs)
if os.path.isfile(batchfile):
    checkpoint = torch.load(batchfile)
    images = checkpoint['images']
    labels = checkpoint['labels']
else:
    images = torch.zeros(args.num_runs, 3, image_size, image_size)
    labels = torch.zeros(args.num_runs).long()
    preds = labels + 1
    while preds.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        for i in list(idx):
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        preds[idx], _ = utils.get_preds(model, images[idx], 'DiabeticRetinopathy', batch_size=args.batch_size, sigma=args.sigma)
    torch.save({'images': images, 'labels': labels}, batchfile)

if args.order == 'rand':
    n_dims = 3 * args.freq_dims * args.freq_dims
else:
    n_dims = 3 * image_size * image_size
if args.num_iters > 0:
    max_iters = int(min(n_dims, args.num_iters))
else:
    max_iters = int(n_dims)
N = int(math.floor(float(args.num_runs) / float(args.batch_size)))
for i in tqdm(range(N)):
    upper = min((i + 1) * args.batch_size, args.num_runs)
    images_batch = images[(i * args.batch_size):upper]
    labels_batch = labels[(i * args.batch_size):upper]
    # replace true label with random target labels in case of targeted attack
    if args.targeted:
        labels_targeted = labels_batch.clone()
        while labels_targeted.eq(labels_batch).sum() > 0:
            labels_targeted = torch.floor(10 * torch.rand(labels_batch.size())).long()
        labels_batch = labels_targeted
    adv, probs, succs, queries, l2_norms, linf_norms, info, attack_images = attacker.simba_batch(
        images_batch, labels_batch, max_iters, args.freq_dims, args.stride, args.epsilon, linf_bound=args.linf_bound,
        order=args.order, targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every,
        seed=args.seed)
    if i == 0:
        all_adv = adv
        all_probs = probs
        all_succs = succs
        all_queries = queries
        all_l2_norms = l2_norms
        all_linf_norms = linf_norms
        all_attack_images = attack_images
    else:
        all_adv = torch.cat([all_adv, adv], dim=0)
        all_probs = torch.cat([all_probs, probs], dim=0)
        all_succs = torch.cat([all_succs, succs], dim=0)
        all_queries = torch.cat([all_queries, queries], dim=0)
        all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
        all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
        all_attack_images = all_attack_images + attack_images
    if args.pixel_attack:
        prefix = 'pixel'
    else:
        prefix = 'dct'
    if args.targeted:
        prefix += '_targeted'
    savefile = '%s/%s_%s_%d_%d_%d_%.4f_%s%s.pth' % (
        args.result_dir, prefix, args.model, args.num_runs, args.num_iters, args.freq_dims, args.epsilon, args.order,
        args.save_suffix)
    torch.save({'adv': all_adv, 'probs': all_probs, 'succs': all_succs, 'queries': all_queries,
                'l2_norms': all_l2_norms, 'linf_norms': all_linf_norms}, savefile)
torch.save(all_attack_images, "./attack_images/DiabeticRetinopathy/train.pth")

# import matplotlib.pyplot as plt
# import numpy as np
# upper = min(args.batch_size, args.num_runs)
# images_batch = images[0:upper]
# labels_batch = labels[0:upper]
# epoch = list(np.arange(0.2, 10, 0.1))
# remaining = []
# prob = []
# for i in epoch:
#     print("第{}轮---------------------------------------".format(i))
#     if args.targeted:
#         labels_targeted = labels_batch.clone()
#         while labels_targeted.eq(labels_batch).sum() > 0:
#             labels_targeted = torch.floor(10 * torch.rand(labels_batch.size())).long()
#         labels_batch = labels_targeted
#     info = attacker.simba_batch(images_batch, labels_batch, max_iters, args.freq_dims, args.stride, i,
#                                 linf_bound=args.linf_bound, order=args.order, targeted=args.targeted,
#                                 pixel_attack=args.pixel_attack, log_every=args.log_every, seed=args.seed)[-1]
#     remaining.append(info.get('remaining'))
#     prob.append(info.get('prob'))
# print("End-----------------------------------------")