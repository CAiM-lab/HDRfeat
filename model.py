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
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

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
        concat3 = self.conv2__(concat3)

        concat3 = torch.concat((F1_2, F2_2, F3_2, concat3), 1)
        concat3 = self.conv2_(concat3)

        F1_i = torch.cat((F1_1, F2_1), 1)
        F1_i = self.att12(F1_i)
        F1_i = self.CBAM(F1_i)
        F1_1 = F1_1 * F1_i + F1_1 


        F3_i = torch.cat((F3_1, F2_1), 1)
        F3_i = self.att12(F3_i)
        F3_i = self.CBAM(F3_i)
        F3_1= F3_1 * F3_i + F3_1

        concat3 = torch.cat((F1_1, F2_1, F3_1, concat3), 1)

        concat3 = self.conv2(concat3)
        F_1 = self.RFDB1(concat3)
        F_2 = self.RFDB2(F_1)
        F_3 = self.RFDB3(F_2)
        output = torch.cat((F_1, F_2, F_3), 1)
        output = self.GFF_1x1(output) 
        output = self.GFF_3x3(output)
        output = output + F2_1
        output = self.conv_up(output)

        output = self.conv3(output)
        output = self.sigmoid(output)

        return output