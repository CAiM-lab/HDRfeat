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

# from unet_model import Unet_SimCLR
from running_func import *
from utils import *
from utils import AverageLoss
from utils import draw_training
from model import HDRfeat

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='HDRfeat')

parser.add_argument('--train-data', default='train.txt')
parser.add_argument('--test_whole_Image', default='./test.txt')
parser.add_argument('--trained_model_dir', default='./trained_model/')
parser.add_argument('--trained_model_filename', default='trained_model16000.pkl')
parser.add_argument('--result_dir', default='./result/')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--restore', default=False)
parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)
parser.add_argument('--batchsize', default=8)
parser.add_argument('--epochs', default=16000)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--save_model_interval', default=1000)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

# parser.add_argument('--norm_G', type=str, default='spectralinstance',
                    # help='instance normalization or batch normalization')

args = parser.parse_args()


torch.manual_seed(args.seed)
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

#load data
train_loaders = torch.utils.data.DataLoader(
    data_loader(args.train_data),
    batch_size=args.batchsize, shuffle=True, num_workers=4)

#make folders of trained model and result
mk_dir(args.result_dir)
mk_dir(args.trained_model_dir)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)


##
model = HDRfeat(args)
model.apply(weights_init_kaiming)
if args.use_cuda:
    model.cuda()


optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
##
start_step = 0
if args.restore and len(os.listdir(args.trained_model_dir)):
    model, start_step = model_restore(model, args.trained_model_dir)
    print('restart from {} step'.format(start_step))

averageLoss = AverageLoss().cuda()
loss_avg_ = []
psnr_avg_ = []
best_psnr_mu = 0

writer = SummaryWriter(log_dir=args.trained_model_dir)

for epoch in range(start_step + 1, args.epochs + 1):
    start = time.time()
    averageLoss.reset()
    train(epoch, model, train_loaders, optimizer, averageLoss, args)
    loss_avg = averageLoss.compute()
    loss_avg_.append(loss_avg.cpu())
    writer.add_scalar('Loss/loss', loss_avg, epoch)
    draw_training(epoch, loss_avg_)
    end = time.time()
    print('epoch:{}, loss: {:.6f}, cost {} seconds'.format(epoch, loss_avg, end - start))
    if epoch % args.save_model_interval == 0:
        model_name = args.trained_model_dir + 'trained_model{}.pkl'.format(epoch)
        torch.save(model.state_dict(), model_name)
