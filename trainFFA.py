import os, sys
import time, math
import argparse, random
from math import exp
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as tfs
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as FF
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torchvision.models import vgg16

from PIL import Image
import matplotlib.pyplot as plt
import warnings

from Datasetloader import RESIDE_Dataset
from Datasetloader import ITS_test_loader, ITS_train_loader

from FFAModel import FFA

steps = 800
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resume = False
eval_step = 400
learning_rate = 0.0001
model_dir = './trained_models_CLEAN/'
trainset = 'its_train'
testset = 'its_test'
network = 'ffa'
gps = 3
blocks = 12
bs = 1
crop = True
crop_size = 240
no_lr_sche = True
model_name = trainset + '_' + network.split('.')[0] + '_' + str(gps) + '_' + str(blocks)
model_dir = model_dir + model_name + '.pk'
log_dir = 'logs/' + model_name
crop_size='whole_img'
if crop:
    crop_size = crop_size


print('log_dir :', log_dir)
print('model_name:', model_name)

models_ = {'ffa': FFA(gps = gps, blocks = blocks)}
loaders_ = {'its_train': ITS_train_loader, 'its_test': ITS_test_loader}
# loaders_ = {'its_train': ITS_train_loader, 'its_test': ITS_test_loader, 'ots_train': OTS_train_loader, 'ots_test': OTS_test_loader}
start_time = time.time()
T = steps

def train_FFA(net, loader_train, loader_test, optim, criterion):
    losses = []
    start_step = 0
    max_ssim = max_psnr = 0
    ssims, psnrs = [], []
    print('Training from scratch *** ')
    for step in range(start_step+1, steps+1):
        net.train()
        lr = learning_rate
        x, y = next(iter(loader_train))
        x = x.to(device); y = y.to(device)
        out = net(x)
        loss = criterion[0](out,y)
        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        print(f'\rtrain loss: {loss.item():.5f} | step: {step}/{steps} | lr: {lr :.7f} | time_used: {(time.time()-start_time)/60 :.1f}',end='',flush=True)
    np.save(f'./numpy_files/{model_name}_{steps}_losses.npy',losses)
    np.save(f'./numpy_files/{model_name}_{steps}_ssims.npy',ssims)
    np.save(f'./numpy_files/{model_name}_{steps}_psnrs.npy',psnrs)
    
    losses, ssims, psnrs = [], [], []

    for step in range(steps):

        losses.append(loss.item())  # assuming loss is calculated each step

        if step % 100 == 0:
            print(f'Step {step}/{steps} | Loss: {loss.item():.4f}')

    torch.save({
        'step': step,
        'losses': losses,
        'ssims': ssims,  # if calculated during training
        'psnrs': psnrs,  # if calculated during training
        'model': net.state_dict()
    }, model_dir)

    # Save numpy arrays for losses, ssims, and psnrs (outside the training loop)
    np.save(f'./numpy_files/{model_name}_{steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{steps}_psnrs.npy', psnrs)

    print(f'Model saved at the end of training at step {step} in {model_dir}')
