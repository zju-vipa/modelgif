import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--model', type=str, default="resnet20")
parser.add_argument('--model-dir', type=str, default=None)
parser.add_argument('--save-freq',
                    type=int,
                    default=2,
                    help='frequence of saving checkpoints')
parser.add_argument('--id', type=str, default='')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('--nc', type=int, default=10, help="num classes")
parser.add_argument('--ul', action="store_true", help="unlearning")
parser.add_argument('--mix-label',
                    action="store_true",
                    help="mix label for unlearning")
parser.add_argument('-g',
                    '--gpu',
                    default=None,
                    type=str,
                    help='GPU id to use.')
arg = parser.parse_args()

if arg.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu

import torch
import numpy as np

seed = arg.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

import torchvision
import model
import train

try:
    architecture = eval(f"model.{arg.model}")
except:
    architecture = eval(f"torchvision.models.{arg.model}")

train_fn = train.TrainFn(arg.lr,
                         arg.batch_size,
                         arg.dataset,
                         architecture,
                         exp_id=arg.id,
                         model_dir=arg.model_dir,
                         save_freq=arg.save_freq,
                         num_class=arg.nc,
                         mix_label=arg.mix_label,
                         unlearning_drop=arg.ul)

for epoch in range(arg.epochs):
    train_fn.train(epoch)

train_fn.validate(arg.dataset, train_fn.net)
