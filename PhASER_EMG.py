
# Utilities
import datetime
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
import sklearn

import random
sys.path.append('./util/')
sys.path.append('./processed_data/')

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
 
from torch.autograd import Function
from scipy.signal import stft, hilbert


## Import custom functions from helper_functions file
from helper_function import set_seed, save_checkpoint
from phaser_models import *


## Import the dataset configs
from dataset_cfg import EMG

##################################################################################################
# Add arguments you want to pass via commandline
##################################################################################################
parser = argparse.ArgumentParser(description='PhASER :  Phase-Augmented Separate Encoding and Residual framework')
parser.add_argument('--log_comment', default='Phase broadcasting for EMG dataset', type=str,
                    metavar='N',
                    )
parser.add_argument('--chkpt_pth', default='./model_chkpt_emg/', type=str,
                    metavar='N',
                    help='which checkpoint do you wanna use to extract embeddings?')

parser.add_argument('--num_epochs', default=50, type=int,
                    metavar='N',
                    )
parser.add_argument('--cuda_pick', default='cuda:5', type=str,
                    metavar='N',
                    )
parser.add_argument('--nperseg_k', default=0.125, type=float,
                    metavar='N',
                    )

parser.add_argument('--model_c', default=1, type=int,
                    metavar='N',
                    )
parser.add_argument('--seed_num', default=2711, type=int,
                    metavar='N',
                    )
parser.add_argument('--scenario', default='S1', type=str,
                    metavar='N',
                    )
parser.add_argument('--dataset_pth', default='/home/payal/TSDG_2023/Processed_Data/EMG/', type=str,
                    metavar='N',
                    help='which checkpoint do you wanna use to extract embeddings?')

args = parser.parse_args()

num_epochs = args.num_epochs
model_chkpt_pth = args.chkpt_pth
log_comment = args.log_comment
cuda_pick = args.cuda_pick
k = args.nperseg_k
c = args.model_c
seed_num = args.seed_num
scenario = args.scenario
data_path = args.dataset_pth


##################################################################################################
set_seed(seed_num)
device = torch.device(cuda_pick if torch.cuda.is_available() else "cpu")
print(device)
##################################################################################################
dataset_cfg = EMG()
##################################################################################################


## If the checkpoint path is not present create it
if not os.path.exists(args.chkpt_pth):
    os.makedirs(args.chkpt_pth)

writer = SummaryWriter()
writer = SummaryWriter('PhASER : EMG')
writer = SummaryWriter(comment=log_comment)

# breakpoint()
##################################################################################################
##************************* Data preparation *****************************************************
##################################################################################################
if (scenario == 'S1') :
    src_dataset  = torch.load(data_path + 'emg_src_env0.pt')
    val_dataset  = torch.load(data_path + 'emg_val_env0.pt')
    trgt_dataset = torch.load(data_path + 'emg_trg_env0.pt')
elif (scenario == 'S2') :
    src_dataset  = torch.load(data_path + 'emg_src_env1.pt')
    val_dataset  = torch.load(data_path + 'emg_val_env1.pt')
    trgt_dataset = torch.load(data_path + 'emg_trg_env1.pt')
elif (scenario == 'S3') :
    src_dataset  = torch.load(data_path + 'emg_src_env2.pt')
    val_dataset  = torch.load(data_path + 'emg_val_env2.pt')
    trgt_dataset = torch.load(data_path + 'emg_trg_env2.pt')
else :
    src_dataset  = torch.load(data_path + 'emg_src_env3.pt')
    val_dataset  = torch.load(data_path + 'emg_val_env3.pt')
    trgt_dataset = torch.load(data_path + 'emg_trg_env3.pt')



class Load_Spectral_Dataset_TSDG(Dataset):
    def __init__(self, dataset_dict, dataset_config, domain_id, hhtAug = False) :
        super().__init__()
        self.num_channels = dataset_config.input_channels
        self.domain_id = domain_id

        # Load samples and labels
        x_data = dataset_dict["samples"]
        y_data = dataset_dict.get("labels")

        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)

        ## Reconstruct an analytic signal using Hilbert transform
        analytic_signal = hilbert(x_data)
        re_hht = np.real(analytic_signal)
        img_hht = np.imag(analytic_signal) 
        
        ## Based on a uniform probability distribution choose the real or imaginary part of the signal
        if hhtAug == True :
            x_aug = img_hht
        else :
            x_aug = re_hht

        
        ## Extract STFT for each channel :: Using the scipy.signal.stft function we can directly process multivariate STFT
        # Calculate the STFT of the signal
        f, t, Zxx = stft(x_aug,
                         fs =dataset_cfg.sampling_rate,
                         nperseg=dataset_cfg.window_size * k,
                         nfft= 1024,
                         )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx) 

        # Normalize data
        if dataset_config.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        else:
            self.transform = None
        
        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.n_samples = x_data.shape[0]
        self.mag = mag
        self.phase = phase

    def __getitem__(self, index):
        mag = self.mag[index]
        phase = self.phase[index]
        mag_phase = np.concatenate((mag, phase), axis=0)
        y = self.y_data[index] if self.y_data is not None else None
        return mag, phase, y, self.domain_id

    def __len__(self):
        return self.n_samples

        


src_dataset_train_re = Load_Spectral_Dataset_TSDG(src_dataset, dataset_cfg, src_dataset['domain_id'], hhtAug = False)
print('Source Dataset size is : ', len(src_dataset_train_re))
src_dataset_train_img = Load_Spectral_Dataset_TSDG(src_dataset, dataset_cfg, src_dataset['domain_id'], hhtAug = True)
print('Source Dataset size is : ', len(src_dataset_train_img))
src_dataset = torch.utils.data.ConcatDataset([src_dataset_train_re, src_dataset_train_img])
print('Source Dataset size is : ', len(src_dataset))
# breakpoint()
vld_dataset= Load_Spectral_Dataset_TSDG(val_dataset, dataset_cfg, val_dataset['domain_id'], hhtAug = False)
trgt_dataset = Load_Spectral_Dataset_TSDG(trgt_dataset, dataset_cfg, trgt_dataset['domain_id'], hhtAug = False)
 
## Dataloaders

train_data_loader = torch.utils.data.DataLoader(dataset=src_dataset, 
                                                batch_size=dataset_cfg.batch_size,
                                                shuffle=dataset_cfg.shuffle, 
                                                drop_last=dataset_cfg.drop_last,
                                                num_workers=2)


valid_dataloader_ood = torch.utils.data.DataLoader(dataset=vld_dataset, 
                                                batch_size=1,
                                                shuffle=dataset_cfg.shuffle, 
                                                drop_last=dataset_cfg.drop_last,
                                                num_workers=1)

target_dataloader = torch.utils.data.DataLoader(dataset=trgt_dataset, 
                                                batch_size=1,
                                                shuffle=dataset_cfg.shuffle, 
                                                drop_last=dataset_cfg.drop_last,
                                                num_workers=1)

##################################################################################################
##************************* Model Design *****************************************************
##################################################################################################
########################################################################################################
# Helper functions for transformers
########################################################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, device, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_len = 33
        # max_len = 376 # FIXME :: UPdeate in the class definitions
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # (L, N, F)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, device):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        x = x + self.pe[:x.size(0)].to(device)
        return self.dropout(x)

  
class encoder(nn.Module):
    def __init__(self, d_model, device): 
        super(encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=10) ## README: d_model is the "f" in forward function of class network
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1) ## num_layers is same as N in the transformer figure in the transformer paper
        self.positional_encoding = PositionalEncoding(device,d_model)
    def forward(self, tgt):
        tgt = self.positional_encoding(tgt, device) ##for positional encoding
        out = self.transformer_encoder(tgt) ##when masking not required, just remove mask=tgt_mask
        return out
    

########################################################################################################
# Model 1: Mag-Phase encoder with magnitude for intermediate broadcasting
########################################################################################################
class phaser_tf(torch.nn.Module):
    def __init__(self, input_channels, num_class, device, c=4, FINnorm=False):
        super(phaser_tf, self).__init__()
        self.lamb = 0.1
        self.device = device
        self.input_channels = input_channels
        c = 10 * c

        

        self.conv1 = nn.Conv2d(input_channels, 2 * c, 5, stride=(2, 2), padding=(2, 2))

        self.ssn1 = SubSpectralNorm(2 * c, 3) #FIXME
        self.ssn2 = SubSpectralNorm(c, 3) #FIXME

        self.conv1_fusion = nn.Conv2d(4*c, 2 * c, 5, stride=(2, 2), padding=(2, 2))

        ### Define the frequency deptwhwise convolution after the fusion feature-set
        self.convfreqdw = nn.Conv2d(2 * c , c, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(c)

        self.freq_dw_conv = nn.Conv2d(
            c,
            c,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=c,
            stride=1,
            dilation=1,
            bias=False,
        )

        self.temp_dw_conv_tf = encoder(c, device) 

        self.bn2 = nn.BatchNorm2d(c)

        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        
        self.convtempdw = nn.Conv2d(c, c, kernel_size=(1, 1), bias=False)
        
        self.conv_magbroadcast = nn.Conv2d( 2 * c, c, 5, stride=(2,2), padding=(2, 2))
        
        self.flag = False
               
        ### Max Pooling
        self.maxpool = nn.MaxPool2d(2)

        self.clfHead = nn.Conv2d(int(c), num_class, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fin = FrequencyInstanceNorm(129) # FIXME 

        self.lastLayer = nn.LogSoftmax(dim=1)

    def forward(self, mag, phase):
        ################################ Mag Feature Encoder ################################
        # print('Shape of mag is : ', np.shape(mag))
        out_m = self.conv1(mag)
        out_m = self.ssn1(out_m)
        # print('Shape of out_m is : ', np.shape(out_m))
        ################################ Phase Feature Encoder ################################
        out_p = self.conv1(phase)
        ################################ Fusion Encoder ################################
        out = torch.cat((out_m, out_p), dim=1)
        out = self.conv1_fusion(out)
        ################################ Call the frequency feature encoder #############
        # f2
        #############################
        # print("TransitionBlock :: Input shape is = ", np.shape(out))
        out = self.convfreqdw(out)
        # print("TransitionBlock :: convfreqdw shape is = ", np.shape(out))
        out = self.bn1(out)
        out = self.relu(out)
        # print("TransitionBlock :: Input to freq_dw_conv shape is = ", np.shape(out))
        out = self.freq_dw_conv(out)
        # print("TransitionBlock :: Output of freq_dw_conv shape is = ", np.shape(out))
        out = self.ssn2(out)
        # print("TransitionBlock :: Output of ssn shape is = ", np.shape(out))
        #############################
        # auxilary = out
        auxilary = self.conv_magbroadcast(out_p)
        # print("TransitionBlock :: Shape of out_mag is = ", np.shape(auxilary))
        out = out.mean(2, keepdim=True)  # frequency average pooling 
       
        out = torch.squeeze(out, dim=2)
        # print("TransitionBlock :: Input to temporal transformer after squeeze is = ", np.shape(out))
        out = out.permute(2,0,1)
        # print("TransitionBlock :: Input to temporal transformer after permute is = ", np.shape(out))
        out = self.temp_dw_conv_tf(out)
        # print("TransitionBlock :: Output of temporal transformer is = ", np.shape(out))
        out = torch.unsqueeze(out, dim=2)
        # print("TransitionBlock :: Output of temporal transformer after unsqueeze is = ", np.shape(out))
        out = out.permute(1,3,2,0)
        # print("TransitionBlock :: Output of temporal transformer after permute is = ", np.shape(out))
        out = self.bn2(out)
        out = self.swish(out)
        # print("TransitionBlock :: Input to conv1x1_2 shape is = ", np.shape(out))
        out = self.convtempdw(out)
        out = self.channel_drop(out)
        # print("TransitionBlock :: Output of conv1x1_2 shape is = ", np.shape(out))
        #############################

        out = auxilary + out
        out = self.relu(out)
        out = self.fin(out)
        # print("TransitionBlock :: Final out shape is = ", np.shape(out))
        # breakpoint()

        out = self.maxpool(out)
        # print("Main Function :: Output of block2_1 is : ", np.shape(out))

        out = self.clfHead(out)
        # print("Main Function :: Output of block7_1 is : ", np.shape(out))
        out = self.avgpool(out)
        # print("Main Function :: Output of block8_1 is : ", np.shape(out))

        clipwise_output = torch.squeeze(torch.squeeze(out, dim=2), dim=2)
       
        clipwise_output = self.lastLayer(clipwise_output)
        return clipwise_output
    

## you may
# class phaser_nontf(torch.nn.Module):
#     def __init__(self, cfg, num_class, device, c=4, FINnorm=False, lastAct=None):
#         super(phaser_nontf, self).__init__()
#         self.lamb = 0.1
#         self.lastAct = lastAct
#         self.device = device
#         c = 10 * c

#         self.conv1 = nn.Conv2d(dataset_cfg.input_channels, 2 * c, 5, stride=(2, 2), padding=(2, 2))
#         self.ssn1 = SubSpectralNorm(2 * c, 3)
#         self.ssn2 = SubSpectralNorm(c, 3) 

#         self.conv1_fusion = nn.Conv2d(4*c, 2 * c, 5, stride=(2, 2), padding=(2, 2))
        

#         self.block1_1 = TransitionBlock(4 * c, c)
#         self.block1_2 = BroadcastedBlock(c)

#         self.conv_magbroadcast = nn.Conv2d( 2 * c, c, 5, stride=(1,1), padding='same')

#         self.block2_1 = nn.MaxPool2d(2)

#         self.block5_1 = TransitionBlock(int(c), int(2 * c))
#         self.block5_2 = BroadcastedBlock(int(2 * c))

#         self.block6_1 = TransitionBlock(int(2 * c), int(2.5 * c))
#         self.block6_2 = BroadcastedBlock(int(2.5 * c))
#         self.block6_3 = BroadcastedBlock(int(2.5 * c))

#         self.block7_1 = nn.Conv2d(int(2.5 * c), num_class, 1)

#         self.block8_1 = nn.AdaptiveAvgPool2d((1, 1))
#         self.norm = FINnorm


    # def forward(self, mag, phase, add_noise=False, training=False, noise_lambda=0.1, k=2):
    #     ################################ Mag Feature Encoder ################################
    #     out_m = self.conv1(mag)
    #     # out_m = self.ssn1(out_m)
    #     ################################ Phase Feature Encoder ################################
    #     out_p = self.conv1(phase)
    #     # out_p = self.ssn1(out_p)
    #     ################################ Fusion Encoder ################################
    #     out = torch.cat((out_m, out_p), dim=1)
    #     # out = self.conv1_fusion(out)
        
    #     out = self.block1_1(out)
    #     out = self.block1_2(out)

    #     ######## Phase residual 1####################
    #     auxilary = self.conv_magbroadcast(out_p)
    #     out = auxilary + out

    #     out = self.block2_1(out)

    #     out = self.block5_1(out)
    #     out = self.block5_2(out)
    #     out = self.block6_1(out)
    #     out = self.block6_2(out)
    #     out = self.block6_3(out)
    #     out = self.block7_1(out)
    #     out = self.block8_1(out)

    #     clipwise_output = torch.squeeze(torch.squeeze(out, dim=2), dim=2)
    #     if self.lastAct == "softmax":
    #         clipwise_output = self.lastLayer(clipwise_output)
    #     return clipwise_output
    
##################################################################################################
##************************* Model Initialisation and Training *****************************************************
##################################################################################################
alpha = 0.0001
model = phaser_tf(device=device, input_channels=dataset_cfg.input_channels, num_class = dataset_cfg.num_classes , c=c, FINnorm=False).to(device)
optimizer =  torch.optim.Adam(model.parameters(), lr = 0.0001)

class_loss_criterion =nn.CrossEntropyLoss()

print('Number of trainable parameters:', sum(p.numel() for p in model.parameters()))


##################################################################################################
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter():
    def __init__(self, num_batches, meters, prefix=""): 
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

        
def train_one_epoch(train_loader, model, class_loss_criterion, optimizer, epoch):
    class_loss_list = np.zeros(len(train_loader)) 
    overall_loss_list = np.zeros(len(train_loader))
    class_acc_list = np.zeros(len(train_loader))

    loss_class = AverageMeter('Class Loss', ':.4f')
    loss_overall = AverageMeter('Overall Loss', ':.4f')
    
    model.train()
    model.zero_grad()
    
    
    for i,(feat_mag, feat_phase, y, d) in enumerate(train_data_loader) : 
        correct = 0
        y = y.to(device)
        d = d.to(device)
        feat_mag = feat_mag.to(device).float()
        feat_phase = feat_phase.to(device).float()
                
        class_output = model(feat_mag, feat_phase)
        
        # loss list of a batch
        loss_class_iter = class_loss_criterion(class_output, y)

        loss = loss_class_iter 
        ## Compute the accuracy per batch
        _, predicted_labels = torch.max(class_output.data, 1)
        correct += predicted_labels.eq(y).sum().item()

        # Average loss of a batch
        curr_loss = loss_overall.update(loss.item(), feat_mag.size(0))
        curr_class_loss = loss_class.update(loss_class_iter.item(), feat_mag.size(0))
        
        curr_class_acc = correct/feat_mag.size(0)
        acc_class_meter = 'Class Acc: {:.4f}'.format(curr_class_acc)

            
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Maintain the list of all losses per iteration (= total data/batch size)
        class_loss_list[i] = curr_class_loss
        overall_loss_list[i] = curr_loss
        class_acc_list[i] = curr_class_acc

        progress = ProgressMeter(
        len(train_loader), # total_data/batch_size
        [loss_overall, loss_class, acc_class_meter],
        prefix="Epoch: [{}]".format(epoch))


        if (i % 50 == 0) | (i == len(train_loader)-1):
            progress.display(i)
        if (i == len(train_loader)-1):
            print('End of Epoch', epoch, 'Overall loss is','%.4f' % np.mean(overall_loss_list), '    Training accuracy is ', '%.4f' % np.mean(class_acc_list))    
    return overall_loss_list, class_loss_list, class_acc_list

def evaluate_one_epoch(valid_loader, model, class_loss_criterion, optimizer, epoch):
    
    with torch.no_grad():
        correct_v = 0
        class_loss_list = np.zeros(len(valid_loader))

        #Gives output per example since batch_size = 1
        for i,(feat_mag, feat_phase, y, d) in enumerate(valid_loader) : 
            model.eval()
            y = y.to(device)
            d = d.to(device)
            feat_mag = feat_mag.to(device).float()
            feat_phase = feat_phase.to(device).float()
                    
            class_output = model(feat_mag, feat_phase)
            
            # loss list of a batch
            class_loss_list[i] = class_loss_criterion(class_output, y)
            
            ## Compute class accuracy
            _, predicted_labels_v = torch.max(class_output.data, 1)
            correct_v += predicted_labels_v.eq(y).sum().item()

        
        class_loss_valid = sum(class_loss_list)/len(class_loss_list)
        class_acc_valid = correct_v/len(class_loss_list)

   
    return class_loss_valid, class_acc_valid


def evaluate_ood_one_epoch(trgt_loader, model, class_loss_criterion):
    ## Assume there is no mini-batch in validation
    ## Batch Size is 1
    with torch.no_grad():
        correct_v = 0
        class_loss_list = np.zeros(len(trgt_loader))

        #Gives output per example since batch_size = 1
        for i,(feat_mag, feat_phase, y, d) in enumerate(trgt_loader) : 
            model.eval()
            y = y.to(device)
            d = d.to(device)
            feat_mag = feat_mag.to(device).float()
            feat_phase = feat_phase.to(device).float()
                    
            class_output = model(feat_mag, feat_phase)
            
            # loss list of a batch
            class_loss_list[i] = class_loss_criterion(class_output, y)
            
            ## Compute class accuracy
            _, predicted_labels_v = torch.max(class_output.data, 1)
            correct_v += predicted_labels_v.eq(y).sum().item()

        
        class_loss_valid = sum(class_loss_list)/len(class_loss_list)
        class_acc_valid = correct_v/len(class_loss_list)
    return class_loss_valid, class_acc_valid

# Initialize variables to track the best model
best_val_ood_acc = 0.0
best_epoch = 0

test_acc_list = np.zeros(num_epochs)
for epoch in range(0, num_epochs):
    print('Inside Epoch : ', epoch)

    # Train for one epoch
    overall_loss_list, class_loss_list, class_acc_train = train_one_epoch(train_data_loader, model, class_loss_criterion, optimizer, epoch)

    # Average loss through all iterations --> Avg loss of an epoch
    overall_loss_epoch = sum(overall_loss_list)/len(overall_loss_list)
    class_loss_epoch = sum(class_loss_list)/len(class_loss_list)
    class_acc_epoch = sum(class_acc_train)/len(class_acc_train)
    
    writer.add_scalar("Overall Loss/train", overall_loss_epoch, epoch) 
    writer.flush()
    writer.add_scalar("Class Loss/train", class_loss_epoch, epoch) 
    writer.flush()
    writer.add_scalar("Accuracy/train", class_acc_epoch, epoch) 
    writer.flush()

    # Evaluate every epoch for out-of-domain data in validation
    class_loss_valid_ood, class_acc_valid_ood = evaluate_ood_one_epoch(valid_dataloader_ood, model, class_loss_criterion)
    writer.add_scalar("Accuracy/valid_ood", class_acc_valid_ood, epoch) 
    writer.flush()
    writer.add_scalar("Class Loss/valid_ood", class_loss_valid_ood, epoch) 
    writer.flush()

    # Save the best model based on validation OOD performance
    if class_acc_valid_ood > best_val_ood_acc:
        best_val_ood_acc = class_acc_valid_ood
        best_epoch = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=model_chkpt_pth + 'best_val_ood_model.pth.tar')

    # Evaluate on target dataset and log accuracy
    class_loss_trg, class_acc_trg = evaluate_ood_one_epoch(target_dataloader, model, class_loss_criterion)
    writer.add_scalar("Accuracy/target", class_acc_trg, epoch) 
    test_acc_list[epoch] = class_acc_trg
    writer.flush()

# After training, load the best model and evaluate on target dataset
# Create a new model instance
best_model = phaser_tf(device=device, input_channels=dataset_cfg.input_channels, 
                       num_class=dataset_cfg.num_classes, c=c, FINnorm=False).to(device)

# Load the best model checkpoint
checkpoint = torch.load(model_chkpt_pth + 'best_val_ood_model.pth.tar')
best_model.load_state_dict(checkpoint['state_dict'])

# Evaluate the best model on the target dataset
class_loss_trg, class_acc_trg = evaluate_ood_one_epoch(target_dataloader, best_model, class_loss_criterion)
print(f'Best Epoch: {best_epoch}')
print(f'Best Validation OOD Accuracy: {best_val_ood_acc}')
print(f'Target Class Accuracy with Best Model: {class_acc_trg}')

writer.close()

# Write the test accuracy to a csv file
df = pd.DataFrame()
df['target_ood'] = pd.DataFrame(test_acc_list)
df.to_csv(str(seed_num) + '_' + scenario + '_emg_acc.csv', index=False, header=True)