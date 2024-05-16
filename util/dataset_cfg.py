"""
-- Contains the dataset configuration for the different datasets
"""


# Utilities
import string
import sys
import time
import matplotlib.pyplot as plt
import IPython.display as ipd
import argparse
import math
from tqdm import tqdm 

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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



## Import custom functions from helper_functions file
from helper_function import set_seed

# eeg scenarios
def __eeg_scenarios__(dataset_cfg, scenario) :

    # Scenario 1: Values 0 to 4
    # Scenario 2: Values 5 to 9
    # Scenario 3: Values 10 to 14
    # Scenario 4: Values 15 to  19
    if (scenario == 'S1'):
        dataset_cfg.src_domains = np.array(range(5,18))
        dataset_cfg.val_domains = np.array(range(18,20))
        dataset_cfg.trg_domains = np.array(range(0,5))
    elif (scenario == 'S2'):
        a = np.array(range(0,5))
        b = np.array(range(10,18))
        dataset_cfg.src_domains = np.concatenate((a,b), axis=0)
        dataset_cfg.val_domains = np.array(range(18,20))
        dataset_cfg.trg_domains = np.array(range(5,10))
    elif (scenario == 'S3'):
        a = np.array(range(0,10))
        b = np.array(range(15,18))
        dataset_cfg.src_domains = np.concatenate((a,b), axis=0)
        dataset_cfg.val_domains = np.array(range(18,20))
        dataset_cfg.trg_domains = np.array(range(10,15))
    else :
        dataset_cfg.src_domains = np.array(range(0,13))
        dataset_cfg.val_domains = np.array(range(13,15))
        dataset_cfg.trg_domains = np.array(range(15,20)) 


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.sampling_rate = 100 
        self.window_size = 128 
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.shuffle = True
        self.drop_last = True
        self.normalize = False
        self.num_domains = 20
        self.batch_size = 16
 
        ## Source     :: 0:12
        ## Validation :: 13:14
        ## Target     :: 15:19
        
        self.src_domains = np.array(range(0,13))
        self.val_domains = np.array(range(13,15))
        self.trg_domains = np.array(range(15,20))

        # # Number of features in the input dataset
        self.input_channels = 1


# WISDM dataset
def __wisdm_scenarios__(dataset_cfg, scenario) :
    # Scenario 1: Values 0 to 8
    # Scenario 2: Values 9 to 17
    # Scenario 3: Values 18 to 26
    # Scenario 4: Values 27 to 35 
    if (scenario == 'S1'):
        dataset_cfg.src_domains = np.array(range(9,32))
        dataset_cfg.val_domains = np.array(range(32,36))
        dataset_cfg.trg_domains = np.array(range(0,10))
    elif (scenario == 'S2'):
        a = np.array(range(5,10))
        b = np.array(range(18,36))
        dataset_cfg.src_domains = np.concatenate((a,b), axis=0)
        dataset_cfg.val_domains = np.array(range(0,5))
        dataset_cfg.trg_domains = np.array(range(10,18))
    elif (scenario == 'S3'):
        a = np.array(range(0,19))
        b = np.array(range(32,36))
        dataset_cfg.src_domains = np.concatenate((a,b), axis=0)
        dataset_cfg.val_domains = np.array(range(28,32))
        dataset_cfg.trg_domains = np.array(range(18,28))
    else :
        dataset_cfg.src_domains = np.array(range(0,23))
        dataset_cfg.val_domains = np.array(range(23,28))
        dataset_cfg.trg_domains = np.array(range(28,36)) 
        
class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.sampling_rate = 20
        self.window_size = 32 
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.num_domains = 36

        ## Source     :: 0:22
        ## Validation :: 23:28
        ## Target     :: 29:35
        self.src_domains = np.array(range(0,23))
        self.val_domains = np.array(range(23,28))
        self.trg_domains = np.array(range(28,35))

        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False 
        self.normalize = False 
        self.batch_size = 32
        # Number of features in the input dataset
        self.input_channels = 3


## HHAR
def __hhar_scenarios__(dataset_cfg, scenario) :
    # Scenario 1: Values 0,1
    # Scenario 2: Values 2,3
    # Scenario 3: Values 4,5
    # Scenario 4: Values 6,7,8
    if (scenario == 'S1'):
        dataset_cfg.src_domains = np.array(range(2,7))
        dataset_cfg.val_domains = np.array(range(7,9))
        dataset_cfg.trg_domains = np.array((0,1))
    elif (scenario == 'S2'):
        a = np.array(range(0,2))
        b = np.array(range(4,7))
        dataset_cfg.src_domains = np.concatenate((a,b), axis=0)
        dataset_cfg.val_domains = np.array(range(7,9))
        dataset_cfg.trg_domains = np.array((2,3))
    elif (scenario == 'S3'):
        a = np.array(range(0,4))
        b = np.array(range(6,7)) 
        dataset_cfg.src_domains = np.concatenate((a,b), axis=0)
        dataset_cfg.val_domains = np.array(range(7,9))
        dataset_cfg.trg_domains = np.array((4,5))
    else :
        dataset_cfg.src_domains = np.array(range(0,4))
        dataset_cfg.val_domains = np.array(range(4,6))
        dataset_cfg.trg_domains = np.array((6,7,8)) 
        
class HHAR(object):
    def __init__(self):
        super(HHAR, self).__init__()
        self.sampling_rate = 100
        self.window_size = 32 
        # self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.sequence_len = 128
        self.num_domains = 36

        self.src_domains = np.array(range(0,6))
        self.val_domains = np.array(range(4,6))
        self.trg_domains = np.array((6,7,8)) 

        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False 
        self.normalize = False 
        self.batch_size = 128
        # Number of features in the input dataset
        self.input_channels = 3


## UCIHAR
def __ucihar_scenarios__(dataset_cfg, scenario) :
    # Scenario 1: Values 0,7
    # Scenario 2: Values 8,15
    # Scenario 3: Values 16,23
    # Scenario 4: Values 24,31
    # Use only in-domain validation
    if (scenario == 'S1'):
        dataset_cfg.src_domains = np.array(range(8,28))
        dataset_cfg.val_domains = np.array(range(28,31))
        dataset_cfg.trg_domains = np.array(range(0,8))
    elif (scenario == 'S2'):
        a = np.array(range(0,7))
        b = np.array(range(16,28))
        dataset_cfg.src_domains = np.concatenate((a,b), axis=0)
        dataset_cfg.val_domains = np.array(range(28,31))
        dataset_cfg.trg_domains = np.array(range(8,16))
    elif (scenario == 'S3'):
        a = np.array(range(0,16))
        b = np.array(range(24,28))
        dataset_cfg.src_domains = np.concatenate((a,b), axis=0)
        dataset_cfg.val_domains = np.array(range(28,31))
        dataset_cfg.trg_domains = np.array(range(16,24))
    else :
        dataset_cfg.src_domains = np.array(range(0,20))
        dataset_cfg.val_domains = np.array(range(20,24))
        dataset_cfg.trg_domains = np.array(range(24,31)) 
        
class UCIHAR(object):
    def __init__(self):
        super(UCIHAR, self).__init__()
        self.sampling_rate = 50
        self.window_size = 32 
        # self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        # self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.num_domains = 30

        self.src_domains = np.array(range(8,28))
        self.val_domains = np.array(range(28,31))
        self.trg_domains = np.array(range(0,8))

        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False 
        self.normalize = False 
        self.batch_size = 128
        # Number of features in the input dataset
        self.input_channels = 9
####################################################################################################
class EMG():
    def __init__(self):
        super(EMG, self).__init__()
        self.sampling_rate = 200 
        self.window_size = 32
        self.num_classes = 6
        self.sequence_len = 200
        self.shuffle = True
        self.drop_last = True
        self.normalize = False
        self.num_domains = 20
        self.batch_size = 128
 
        ## Source     :: 0:12
        ## Validation :: 13:14
        ## Target     :: 15:19
        
        # self.src_domains = np.array(range(0,13))
        # self.val_domains = np.array(range(13,15))
        # self.trg_domains = np.array(range(15,20))

        # # Number of features in the input dataset
        self.input_channels = 8
