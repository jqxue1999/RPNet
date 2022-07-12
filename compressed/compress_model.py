import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from compressed import utils
import models
import torch
import torch.backends.cudnn as cudnn
from compressed import compress
from compressed.utils import get_dataset

parser = argparse.ArgumentParser(description='Model Compress')
parser.add_argument('--model', required=True, type=str, help='mdoel name')
parser.add_argument('--model_dir', required=True, type=str, help='mdoel dir')
parser.add_argument('--dataset', required=True, type=str, help='dataset name')
parser.add_argument('--dataset_dir', default='./data', type=str, help='dataset dir')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--act_bits', default=16, type=int, help='act bits')
parser.add_argument('--weight_bits', default=16, type=int, help='weight bits')
parser.add_argument('--save_dir', required=True, type=str, help='save dir')
parser.add_argument('--input_scale', default=1, type=int, help='input scale')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load model
utils.ensure_dir(args.model_dir)
model_raw = getattr(getattr(models, args.dataset), args.model)().to(device)

checkpoint = torch.load(args.model_dir)
model_raw.load_state_dict(checkpoint['net'])

if device == 'cuda':
    model_raw = torch.nn.DataParallel(model_raw)
    cudnn.benchmark = True

# load dataset
test_dataloader = get_dataset(args.dataset, args.dataset_dir, args.batch_size)

# compress
utils.test(model_raw, test_dataloader)
model_new = compress.CompressedModel(model_raw, input_scale=args.input_scale, act_bits=args.act_bits,
                                     weight_bits=args.weight_bits)
utils.test(model_new, test_dataloader)
model_new.quantize_params()
utils.test(model_new, test_dataloader)

print(model_new)
model_new = torch.nn.DataParallel(model_new)
torch.save(model_new, args.save_dir)