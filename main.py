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


from trainFFA import train_FFA
from trainFFA import models_
from trainFFA import loaders_
from trainFFA import start_time

from CycleGAN.primary_train import train_cycle_gan
from CycleGAN.fine_tune import fine_tune_cycle_gan
from CycleGAN.dataloaders import paired_loader, unpaired_loader
from CycleGAN.ganloss import CycleGANLoss
from CycleGAN.generator_discriminator import generator_XY, generator_YX, discriminator_X, discriminator_Y, optimizer_G, optimizer_D
from CycleGAN.ffamodelredefined import ffa_model


warnings.filterwarnings('ignore')
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

if not os.path.exists('trained_models_CLEAN'):
    os.mkdir('trained_models_CLEAN')
if not os.path.exists('numpy_files'):
    os.mkdir('numpy_files')
if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('samples'):
    os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
    os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    
crop_size='whole_img'
if crop:
    crop_size = crop_size

loader_train = loaders_[trainset]
loader_test = loaders_[testset]
net = models_[network]
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = []
criterion.append(nn.L1Loss().to(device))
optimizer = optim.Adam(params = filter(lambda x: x.requires_grad, net.parameters()), lr=learning_rate, betas=(0.9,0.999), eps=1e-08)
optimizer.zero_grad()

train_FFA(net, loader_train, loader_test, optimizer, criterion)

train_cycle_gan(paired_loader, 
                unpaired_loader, 
                generator_XY, 
                generator_YX, 
                discriminator_X, 
                discriminator_Y, 
                optimizer_G, 
                optimizer_D, 
                criterion, 
                ffa_model, 
                epochs=100)

fine_tune_cycle_gan(paired_loader, 
                    unpaired_loader, 
                    generator_XY, 
                    generator_YX, 
                    discriminator_X, 
                    discriminator_Y, 
                    optimizer_G, 
                    optimizer_D, criterion, 
                    ffa_model, epochs=100, num_paired=25)  # We can change num_paired as needed

