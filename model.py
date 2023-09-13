import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from RFDB.block import RFDB 
from attention_module.cbam import CBAM
import torchsummary
'''
In this model, we have CBAM (with residual), three encoder layers (first layer with CBAM), RFDB
'''

class HDRfeat(nn.Module):
    def __init__(self, args):
        super(HDRfeat, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        self.args = args

        # F-1
        self.conv_enc1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv_enc2 = nn.Conv2d(nFeat, nFeat *2 , kernel_size=3, padding=1, bias=True)
        self.conv_enc3 = nn.Conv2d(nFeat * 2, nFeat * 3, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat * 4, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2_ = nn.Conv2d(nFeat * 7, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2__ = nn.Conv2d(nFeat * 9, nFeat, kernel_size=3, padding=1, bias=True)

        self.att12 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # CBAM module
        self.CBAM = CBAM(nFeat)
        # RFDB
        self.RFDB1 = RFDB(nFeat)
        self.RFDB2 = RFDB(nFeat)
        self.RFDB3 = RFDB(nFeat)
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        # self.conv_down_concat = nn.Conv2d(nFeat*4, nFeat, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2, x3):
        F1_1 = self.relu(self.conv_enc1(x1))
        F2_1 = self.relu(self.conv_enc1(x2))
        F3_1 = self.relu(self.conv_enc1(x3))

        F1_2 = self.relu(self.conv_enc2(F1_1))
        F2_2 = self.relu(self.conv_enc2(F2_1))
        F3_2 = self.relu(self.conv_enc2(F3_1))

        F1_3 = self.relu(self.conv_enc3(F1_2))
        F2_3 = self.relu(self.conv_enc3(F2_2))
        F3_3 = self.relu(self.conv_enc3(F3_2))

        concat3 = torch.concat((F1_3, F2_3, F3_3), 1)
        F_fuse3 = self.conv2__(concat3)

        concat2 = torch.concat((F1_2, F2_2, F3_2, F_fuse3), 1)
        F_fuse2 = self.conv2_(concat2)

        F1_i = torch.cat((F1_1, F2_1), 1)
        F1_i_d = self.att12(F1_i)
        F1_A = self.CBAM(F1_i_d)
        F1_ = F1_1 * F1_A + F1_1 


        F3_i = torch.cat((F3_1, F2_1), 1)
        F3_i_d = self.att12(F3_i)
        F3_A = self.CBAM(F3_i_d)
        F3_ = F3_1 * F3_A + F3_1

        F_ = torch.cat((F1_, F2_1, F3_, F_fuse2), 1)

        F_0 = self.conv2(F_)
        F_1 = self.RFDB1(F_0)
        F_2 = self.RFDB2(F_1)
        F_3 = self.RFDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF) 
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_1
        us = self.conv_up(FDF)

        output = self.conv3(us)
        output = nn.functional.sigmoid(output)

        return output