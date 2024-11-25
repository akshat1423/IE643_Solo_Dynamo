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

from Metrics import ssim 
from Metrics import psnr 

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



def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims, psnrs = [], []
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(device); targets = targets.to(device)
        pred = net(inputs)
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)

    return np.mean(ssims) ,np.mean(psnrs)
