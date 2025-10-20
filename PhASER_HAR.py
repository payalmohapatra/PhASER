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
from scipy.signal import stft, hilbert

from phaser_models import *


## Import custom functions from helper_functions file
from helper_function import set_seed, save_checkpoint


## Import the dataset configs
from dataset_cfg import WISDM, __wisdm_scenarios__, HHAR, __hhar_scenarios__, UCIHAR, __ucihar_scenarios__

##################################################################################################
# Add arguments you want to pass via commandline
##################################################################################################
parser = argparse.ArgumentParser(description='TSDG')
parser.add_argument('--log_comment', default='TSDG:: Phase broadcasting UCIHAR dataset', type=str,
                    metavar='N',
                    )
parser.add_argument('--chkpt_pth', default='model_chkpt/', type=str,
                    metavar='N',
                    help='which checkpoint do you wanna use to extract embeddings?')

parser.add_argument('--num_epochs', default=10, type=int,
                    metavar='N',
                    )
parser.add_argument('--cuda_pick', default='cuda:5', type=str,
                    metavar='N',
                    )
parser.add_argument('--nperseg_k', default=0.125, type=float,
                    metavar='N',
                    )
parser.add_argument('--scenario', default='S1', type=str,
                    metavar='N',
                    )
parser.add_argument('--model_c', default=4, type=int,
                    metavar='N',
                    )
parser.add_argument('--seed_num', default=2711, type=int,
                    metavar='N',
                    )
parser.add_argument('--oot', default=0, type=int,
                    metavar='Apply when you want to do one-to-x generalization.',
                    )
parser.add_argument('--har_type', default='HHAR', type=str,
                    metavar='N',
                    )

parser.add_argument('--dataset_pth', default='/home/payal/TSDG_2023/Raw_Data/WISDM/WISDM/', type=str,
                    metavar='N',
                    help='path to your dataset folder.')


args = parser.parse_args()

num_epochs = args.num_epochs
model_chkpt_pth = args.chkpt_pth
log_comment = args.log_comment
cuda_pick = args.cuda_pick
k = args.nperseg_k
c = args.model_c
scenario = args.scenario
seed_num = args.seed_num
har_type = args.har_type
data_path = args.dataset_pth
oot = args.oot



##################################################################################################
set_seed(seed_num)
device = torch.device(cuda_pick if torch.cuda.is_available() else "cpu")
print(device)
##################################################################################################
# List all the parameter that need update here so that you can make an argparse later
if har_type == 'HHAR' :
    dataset_cfg = HHAR()
    __hhar_scenarios__(dataset_cfg, scenario)
elif har_type == 'UCIHAR' :
    dataset_cfg = UCIHAR()
    __ucihar_scenarios__(dataset_cfg, scenario)
elif har_type == 'WISDM' :
    dataset_cfg = WISDM()
    __wisdm_scenarios__(dataset_cfg, scenario)
elif har_type == 'HHAR_one_to_x' :
    dataset_cfg = HHAR()
    if (oot == 0) :
        dataset_cfg.src_domains = np.array([0])
        dataset_cfg.trg_domains = np.array(range(1,9))
        dataset_cfg.val_domains = np.array([0]) ## dont really need a valid here
    elif (oot == 1) :
        dataset_cfg.src_domains = np.array([1])
        dataset_cfg.trg_domains = np.array([0,2,3,4,5,6,7,8])
        dataset_cfg.val_domains = np.array([1]) ## dont really need a valid here
    elif (oot == 2) :
        dataset_cfg.src_domains = np.array([2])
        dataset_cfg.trg_domains = np.array([0,1,3,4,5,6,7,8])
        dataset_cfg.val_domains = np.array([0]) ## dont really need a valid here
    elif (oot == 3) :
        dataset_cfg.src_domains = np.array([3])
        dataset_cfg.trg_domains = np.array([0,1,2,4,5,6,7,8])
        dataset_cfg.val_domains = np.array([0]) ## dont really need a valid here
    elif (oot == 4) :
        dataset_cfg.src_domains = np.array([4])
        dataset_cfg.trg_domains = np.array([0,1,2,3,5,6,7,8])
        dataset_cfg.val_domains = np.array([0]) ## dont really need a valid here
    elif (oot == 5) :
        dataset_cfg.src_domains = np.array([5])
        dataset_cfg.trg_domains = np.array([0,1,2,3,4,6,7,8])
        dataset_cfg.val_domains = np.array([0]) ## dont really need a valid here
    elif (oot == 6) :
        dataset_cfg.src_domains = np.array([6])
        dataset_cfg.trg_domains = np.array([0,1,2,3,4,5,7,8])
        dataset_cfg.val_domains = np.array([0]) ## dont really need a valid here
    elif (oot == 7) :
        dataset_cfg.src_domains = np.array([7])
        dataset_cfg.trg_domains = np.array([0,1,2,3,4,5,6,8])
        dataset_cfg.val_domains = np.array([0]) ## dont really need a valid here
    elif (oot == 8) :
        dataset_cfg.src_domains = np.array([8])
        dataset_cfg.trg_domains = np.array([0,1,2,3,4,5,6,7])
        dataset_cfg.val_domains = np.array([0]) ## dont really need a valid here



print('HAR dataset is : ', har_type)
print('Scenario is : ', scenario)
print('Target domains are : ', dataset_cfg.trg_domains)
print('Source domains are : ', dataset_cfg.src_domains)
print('Valid domains are : ', dataset_cfg.val_domains)

##################################################################################################


## If the checkpoint path is not present create it
if not os.path.exists(args.chkpt_pth):
    os.makedirs(args.chkpt_pth)

writer = SummaryWriter()
writer = SummaryWriter('TSDG:STFT wo DANN')
writer = SummaryWriter(comment=log_comment)

##################################################################################################
##************************* Data preparation *****************************************************
##################################################################################################

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
        y = self.y_data[index] if self.y_data is not None else None
        return mag, phase, y, self.domain_id

    def __len__(self):
        return self.n_samples

        



def __create_unified_dataset__(domain_cfg, data_path, train_val_test_flag) :
    dataset_train = []
    dataset_test = []
    if train_val_test_flag == 0 : ## train_mode
        domain_id_list = domain_cfg.src_domains
    if train_val_test_flag == 1 : ## val_mode
        domain_id_list = domain_cfg.val_domains
    if train_val_test_flag == 2 : ## test_mode
        domain_id_list = domain_cfg.trg_domains

    for i,_ in enumerate(domain_id_list) :
        if train_val_test_flag == 0:
            print('Working on source domain : ', domain_id_list[i])
            dataset_file_train = torch.load(os.path.join(data_path, "train_" + str(domain_id_list[i]) + ".pt"))
            dataset_file_test  = torch.load(os.path.join(data_path, "test_" + str(domain_id_list[i]) + ".pt"))

            dataset_train_tmp_re = Load_Spectral_Dataset_TSDG(dataset_file_train, domain_cfg, domain_id_list[i], hhtAug = False)
            dataset_train_tmp_img = Load_Spectral_Dataset_TSDG(dataset_file_train, domain_cfg, domain_id_list[i], hhtAug = True)
            dataset_test_tmp  = Load_Spectral_Dataset_TSDG(dataset_file_test, domain_cfg, domain_id_list[i], hhtAug = False)
            ## concatenate the datasets
            dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_train_tmp_re, dataset_train_tmp_img])
            dataset_test  = torch.utils.data.ConcatDataset([dataset_test, dataset_test_tmp])
        if train_val_test_flag == 1:
            print('Working on valid domain : ', domain_id_list[i])
            dataset_file_train = torch.load(os.path.join(data_path, "train_" + str(domain_id_list[i]) + ".pt"))
            dataset_file_test  = torch.load(os.path.join(data_path, "test_" + str(domain_id_list[i]) + ".pt"))

            dataset_train_tmp = Load_Spectral_Dataset_TSDG(dataset_file_train, domain_cfg, domain_id_list[i])
            dataset_test_tmp  = Load_Spectral_Dataset_TSDG(dataset_file_test, domain_cfg, domain_id_list[i])
            ## concatenate the datasets
            dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_train_tmp])
            dataset_test  = torch.utils.data.ConcatDataset([dataset_test, dataset_test_tmp])
        if train_val_test_flag == 2 :
            print('Working on target domain : ', domain_id_list[i])
            dataset_file_train = torch.load(os.path.join(data_path, "train_" + str(domain_id_list[i]) + ".pt"))
            dataset_file_test  = torch.load(os.path.join(data_path, "test_" + str(domain_id_list[i]) + ".pt"))

            dataset_train_tmp = Load_Spectral_Dataset_TSDG(dataset_file_train, domain_cfg, domain_id_list[i])
            dataset_test_tmp  = Load_Spectral_Dataset_TSDG(dataset_file_test, domain_cfg, domain_id_list[i])
            ## concatenate the datasets
            dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_train_tmp])
            dataset_test  = torch.utils.data.ConcatDataset([dataset_test, dataset_test_tmp])
        
        
    
    if train_val_test_flag == 0 :
        # Use this test-set in source domains as a validation set for class based accuracy and model selection
        # Then use the validation set from non-overlapping domains for overall model selection with DANN
        return dataset_train, dataset_test
    else :
        valid_dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
        return valid_dataset, dataset_test ## In this case the second argument is dummy 


src_dataset_train, src_dataset_test = __create_unified_dataset__(dataset_cfg, data_path, 0)
vld_dataset, _ = __create_unified_dataset__(dataset_cfg, data_path, 1)
trgt_dataset, _ = __create_unified_dataset__(dataset_cfg, data_path, 2)
 
## Dataloaders

train_data_loader = torch.utils.data.DataLoader(dataset=src_dataset_train, 
                                                batch_size=dataset_cfg.batch_size*2,
                                                shuffle=dataset_cfg.shuffle, 
                                                drop_last=dataset_cfg.drop_last,
                                                num_workers=1)

valid_dataloader_id = torch.utils.data.DataLoader(dataset=src_dataset_test,
                                               batch_size=1,
                                               shuffle=dataset_cfg.shuffle, 
                                               drop_last=dataset_cfg.drop_last,
                                               num_workers=1)

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

class phaser_nontf(torch.nn.Module):
    def __init__(self, cfg, num_class, device, c=4, FINnorm=False, lastAct=None):
        super(phaser_nontf, self).__init__()
        self.lamb = 0.1
        self.lastAct = lastAct
        self.device = device
        c = 10 * c

        self.conv1 = nn.Conv2d(dataset_cfg.input_channels, 2 * c, 5, stride=(2, 2), padding=(2, 2))
        self.ssn1 = SubSpectralNorm(2 * c, 3) 
        self.ssn2 = SubSpectralNorm(c, 3) 

        self.conv1_fusion = nn.Conv2d(4*c, 2 * c, 5, stride=(1, 1), padding='same')
        

        self.block1_1 = TransitionBlock(2 * c, c)
        self.block1_2 = BroadcastedBlock(c)

        self.conv_magbroadcast = nn.Conv2d( 2 * c, c, 5, stride=(1,1), padding='same')

        self.block2_1 = nn.MaxPool2d(2)

        self.block5_1 = TransitionBlock(int(c), int(2 * c))
        self.block5_2 = BroadcastedBlock(int(2 * c))

        self.conv_magbroadcast2 = nn.Sequential(nn.Conv2d( 2 * c, 2 * c, 5, stride=(1,1), padding='same'),
                                                nn.MaxPool2d(2)
        )

        self.block6_1 = TransitionBlock(int(2 * c), int(2.5 * c))
        self.block6_2 = BroadcastedBlock(int(2.5 * c))
        self.block6_3 = BroadcastedBlock(int(2.5 * c))

        self.block7_1 = nn.Conv2d(int(2.5 * c), num_class, 1)

        self.block8_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = FINnorm
        self.relu = nn.ReLU(inplace=True)


    def forward(self, mag, phase, add_noise=False, training=False, noise_lambda=0.1, k=2):
        ################################ Mag Feature Encoder ################################
        out_m = self.conv1(mag)
        out_m = self.relu(out_m)
        out_m = self.ssn1(out_m)
    
        ################################ Phase Feature Encoder ################################
        out_p = self.conv1(phase)
        out_p = self.relu(out_p)
        out_p = self.ssn1(out_p)
        ################################ Fusion Encoder ################################
        out = torch.cat((out_m, out_p), dim=1)
        out = self.conv1_fusion(out)
        out = self.relu(out)
        
        out = self.block1_1(out)
        out = self.block1_2(out)

        ######## Phase residual 1####################
        auxilary = self.conv_magbroadcast(out_p)
        auxilary = self.relu(auxilary)
        out = auxilary + out
        
        out = self.block2_1(out)

        out = self.block5_1(out)
        out = self.block5_2(out)
        
        # ######## Phase residual 2####################
        auxilary2 = self.conv_magbroadcast2(out_p)
        auxilary2 = self.relu(auxilary2)
        out = auxilary2 + out

        out = self.block6_1(out)
        out = self.block6_2(out)
        out = self.block6_3(out)
        
        out = self.block7_1(out)
        
        out = self.block8_1(out)

        clipwise_output = torch.squeeze(torch.squeeze(out, dim=2), dim=2)
        if self.lastAct == "softmax":
            clipwise_output = self.lastLayer(clipwise_output)
        return clipwise_output
    

##################################################################################################
##************************* Model Initialisation and Training *****************************************************
##################################################################################################
alpha = 0.0001
model = phaser_nontf(device=device, cfg=dataset_cfg, num_class = dataset_cfg.num_classes , c=c, lastAct=None).to(device)
optimizer =  torch.optim.Adam(model.parameters(), lr = 0.0001)

class_loss_criterion =nn.CrossEntropyLoss()

print('Number of trainable parameters:', sum(p.numel() for p in model.parameters()))
##################################################################################################
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
    ## Assume there is no mini-batch in validation
    ## Batch Size is 1
    
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

test_acc_list = np.zeros(num_epochs)
train_acc_list = np.zeros(num_epochs)
val_id_acc_list = np.zeros(num_epochs)
# Initialize variables to track the best model
best_val_ood_acc = 0.0
best_epoch = 0

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

    # Evaluate every epoch for in-domain data in validation
    class_loss_valid, class_acc_valid = evaluate_one_epoch(valid_dataloader_id, model, class_loss_criterion, optimizer, epoch)
    val_id_acc_list[epoch] = class_acc_valid
    writer.add_scalar("Accuracy/valid_id", class_acc_valid, epoch) 
    writer.flush()
    writer.add_scalar("Class Loss/valid_id", class_loss_valid, epoch) 
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

    # Evaluate on target dataset and log accuracy --> Only for logging purpose (saving only using the best val ood acc  model)
    class_loss_trg, class_acc_trg = evaluate_ood_one_epoch(target_dataloader, model, class_loss_criterion)
    writer.add_scalar("Accuracy/target", class_acc_trg, epoch) 
    test_acc_list[epoch] = class_acc_trg
    writer.flush()

# After training, load the best model and evaluate on target dataset
# Create a new model instance
best_model = phaser_nontf(device=device, cfg=dataset_cfg, num_class=dataset_cfg.num_classes, c=c, lastAct=None).to(device)

# Load the best model checkpoint
checkpoint = torch.load(model_chkpt_pth + 'best_val_ood_model.pth.tar')
best_model.load_state_dict(checkpoint['state_dict'])

# Evaluate the best model on the target dataset
class_loss_trg, class_acc_trg = evaluate_ood_one_epoch(target_dataloader, best_model, class_loss_criterion)
print(f'Best Epoch: {best_epoch}')
print(f'Best Validation OOD Accuracy: {best_val_ood_acc}')
print(f'Target Class Accuracy with Best Model: {class_acc_trg}')

# Write the test accuracy to a csv file
df = pd.DataFrame()
df['target_ood'] = pd.DataFrame(test_acc_list)
df['val_id'] = val_id_acc_list
if (har_type == 'HHAR_one_to_x'):
    df.to_csv(har_type + '_' + str(seed_num) + '_' + str(oot) + '.csv', index=False, header=True)
else:
    df.to_csv(har_type + '_' + str(seed_num) + '_' + scenario + '.csv', index=False, header=True)
