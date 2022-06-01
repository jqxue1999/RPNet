import argparse
from compressed import utils
import models
import torch
import torchvision
import torchvision.transforms as transforms
from compressed import compress

parser = argparse.ArgumentParser(description='Model Compress')
parser.add_argument('--model_name', required=True, type=str, help='mdoel name')
parser.add_argument('--model_dir', required=True, type=str, help='mdoel dir')
parser.add_argument('--dataset_dir', required=True, type=str, help='dataset_dir')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--act_bits', default=16, type=int, help='act bits')
parser.add_argument('--weight_bits', default=16, type=int, help='weight bits')
parser.add_argument('--save_dir', required=True, type=str, help='save dir')
parser.add_argument('--input_scale', default=1, type=int, help='input scale')
args = parser.parse_args()

# load model
utils.ensure_dir(args.model_dir)
model_raw = getattr(models, args.model_name)().cuda()
model_raw = torch.nn.DataParallel(model_raw)
checkpoint = torch.load(args.model_dir)
model_raw.load_state_dict(checkpoint['net'])

# load dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_data = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

utils.test(model_raw, test_dataloader)
model_new = compress.CompressedModel(model_raw, input_scale=args.input_scale, act_bits=args.act_bits, weight_bits=args.weight_bits)
utils.test(model_new, test_dataloader)
model_new.quantize_params()
utils.test(model_new, test_dataloader)

print(model_new)
model_new = torch.nn.DataParallel(model_new)
torch.save(model_new, args.save_dir)