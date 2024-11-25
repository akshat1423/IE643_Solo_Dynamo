import os
import random
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image
import torch
import itertools
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
from PIL import Image
from torch.cuda.amp import GradScaler, autocast  # For mixed precision

from FFAModel import FFA

class GeneratorFFA(FFA):
    def __init__(self, gps=3, blocks=12):
        super(GeneratorFFA, self).__init__(gps=gps, blocks=blocks)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

generator_XY = GeneratorFFA(gps=3, blocks=12)  # Hazy -> Clean
generator_YX = GeneratorFFA(gps=3, blocks=12)  # Clean -> Hazy
discriminator_X = Discriminator()  # Discriminator for clean images
discriminator_Y = Discriminator()
optimizer_G = optim.Adam(itertools.chain(generator_XY.parameters(), generator_YX.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(itertools.chain(discriminator_X.parameters(), discriminator_Y.parameters()), lr=0.0002, betas=(0.5, 0.999))

