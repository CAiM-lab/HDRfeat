# adapted from https://github.com/qingsenyangit/AHDRNet with revision
import torch
import numpy as np
import time
import argparse
import torch.optim as optim
import torch.utils.data
import scipy.io as scio
from torch.nn import init
from dataset import DatasetFromHdf5

from model import HDRfeat
from running_func import *
from utils import *

parser = argparse.ArgumentParser(description='HDRfeat')

parser.add_argument('--train-data', default='train.txt')
parser.add_argument('--test_whole_Image', default='./test.txt')
parser.add_argument('--trained_model_dir', default='./')
parser.add_argument('--trained_model_filename', default='HDRfeat_model.pkl')
parser.add_argument('--result_dir', default='./result/')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--restore', default=False)
parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)
parser.add_argument('--batchsize', default=1)
parser.add_argument('--epochs', default=16000)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--save_model_interval', default=5)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

args = parser.parse_args()


torch.manual_seed(args.seed)
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

#load data
test_loaders = torch.utils.data.DataLoader(
    testimage_dataloader(args.test_whole_Image),
    batch_size=1, shuffle=False, num_workers=1)


model = HDRfeat(args)

model = model_restore_test(model, args.trained_model_dir, args.trained_model_filename)
if args.use_cuda:
    model.cuda()

testing_fun(model, test_loaders, args)