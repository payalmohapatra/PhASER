# Utilities
import string
import sys
import time
import matplotlib.pyplot as plt
import IPython.display as ipd
import argparse
import math
from tqdm import tqdm 
#from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import sklearn

import random
## General pytorch libraries
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

## Import audio related packages
from scipy.signal import stft, hilbert


class TransitionBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        dilation=1,
        stride=1,
        temp_pad=(0, 1),
    ):
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(
            planes,
            planes,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=planes,
            stride=stride,
            dilation=dilation,
            bias=False,
        )
        self.ssn = nn.BatchNorm2d(planes)
    
        self.temp_dw_conv = nn.Conv2d(
            planes,
            planes,
            kernel_size=(1, 3),
            padding=temp_pad,
            groups=planes,
            dilation=dilation,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)
        self.flag = False
        
        # attention layer with learnable parameters
        self.cnn_q = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False, padding=(0,0))
        self.cnn_k = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False, padding=(0,0))
        self.cnn_v = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False, padding=(0,0))

    def forward(self, x):
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling 
        
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################
        # breakpoint()

        out = auxilary + out
        out = self.relu(out)
        return out

class BroadcastedBlock(nn.Module):
    def __init__(
        self,
        planes: int,
        dilation=1,
        stride=1,
        temp_pad=(0, 1),
    ):
        super(BroadcastedBlock, self).__init__()
        self.freq_dw_conv = nn.Conv2d(
            planes,
            planes,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=planes,
            dilation=dilation,
            stride=stride,
            bias=False,
        )
        self.ssn1 = nn.BatchNorm2d(planes)
        self.temp_dw_conv = nn.Conv2d(
            planes,
            planes,
            kernel_size=(1, 3),
            padding=temp_pad,
            groups=planes,
            dilation=dilation,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling
        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out

class FrequencyInstanceNorm(nn.Module):
    def __init__(self, num_freq):
        super(FrequencyInstanceNorm, self).__init__()
        self.num_freq = num_freq
        self.norm = nn.InstanceNorm2d(num_freq)

    def forward(self, x):
        out = torch.permute(x, (0, 2, 1, 3))
        out = self.norm(out)
        out = torch.permute(out, (0, 2, 1, 3))
        return out

class SubSpectralNorm(nn.Module):
    def __init__(self, C, S=1, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C * S)
        self.bn_naive = nn.BatchNorm2d(C)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        # print('SSN : F is ', F)
        if (F % self.S == 0) :
            x = x.view(N, C * self.S, F // self.S, T)
            x = self.bn(x)
        else :
            # Take a batch norm
            x = self.bn_naive(x)
        
        return x.view(N, C, F, T)