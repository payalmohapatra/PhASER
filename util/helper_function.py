import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd
import math
from tqdm import tqdm

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
import sklearn
from torch.utils.tensorboard import SummaryWriter
import random
from torch.autograd import Function


##################################################################################################
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
##################################################################################################
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
##################################################################################################


##################################################################################################
# **************** Different learning rate schedulers ******************************************** 
##################################################################################################
def adjust_learning_rate_cosine_anealing(optimizer, init_lr, epoch, num_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    print('Learning rate inside adjusting cosine lr = ', cur_lr)

def adjust_learning_rate_warmup_time(optimizer, init_lr, epoch, num_epochs, model_size, warmup):
    """Decay the learning rate based on warmup schedule based on time
    Source :: https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer 
    """
    cur_lr = (model_size ** (-0.5) * min((epoch+1) ** (-0.5), (epoch+1) * warmup ** (-1.5))) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    print('Learning rate inside adjusting warmup decay lr = ', cur_lr)

def naive_lr_decay(optimizer, init_lr, epoch, num_epochs):
    """
    Make 3 splits in the num_epochs and just use that to decay the lr 
    """
    if (epoch < np.ceil(num_epochs/4)) :
        cur_lr = init_lr
    elif (epoch < np.ceil(num_epochs/2)) :
        cur_lr = 0.5 * init_lr
    else :
        cur_lr = 0.25 * init_lr    

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    print('Learning rate inside naive decay lr = ', cur_lr)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)