
#In case if torch summary is not installed, pip install torchsummary

# imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torch import nn
from torchsummary import summary


def show_tensor_images(tensor_img, num_images=16, size=(1, 28, 28)):
    """
    Plot a grid of tensor images (assumes images in range [-1,1]).
    """
    unflat_img = tensor_img[:num_images].detach().cpu()
    unflat_img = unflat_img * 0.5 + 0.5    # DENORMALIZE back to [0,1]
    img_grid = make_grid(unflat_img, nrow=int(np.sqrt(num_images)))
    plt.figure(figsize=(4,4))
    plt.imshow(img_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
    return 

def get_disc_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            get_disc_block(1, 64, 4, 2, 1, use_bn=False),   # -> (64,14,14)
            get_disc_block(64, 128, 4, 2, 1),               # -> (128,7,7)
            get_disc_block(128, 256, 4, 2, 1),              # -> (256,3,3)
        )
        # compute flattened feature size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            feat = self.conv(dummy)
            self._flat_dim = feat.view(1, -1).size(1)

        self.fc = nn.Linear(self._flat_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).view(-1, 1)
    
    
def get_gen_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, final_block=False):
    if final_block:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Tanh()
        )
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    """
    DCGAN-style generator mapping (noice_dim) -> (1,28,28)
    We design the transpose conv chain so final output is 28x28.
    """
    def __init__(self, noice_dim=100):
        super(Generator, self).__init__()
        self.noice_dim = noice_dim
        # Start from (noice_dim,1,1) -> project into feature map
        # We'll use a small "project" layer via ConvTranspose to get upsampled maps
        self.net = nn.Sequential(
            # input: (B, noice_dim, 1, 1)
            get_gen_block(noice_dim, 256, 4, 1, 0),   # -> (256,4,4)
            get_gen_block(256, 128, 4, 2, 1),         # -> (128,8,8)
            get_gen_block(128, 64, 4, 2, 1),          # -> (64,16,16)
            get_gen_block(64, 1, 4, 2, 1, final_block=True) # -> (1,32,32) -> we'll crop/pad to 28
        )
        # Because the straightforward chain gives 32x32, we'll center-crop to 28x28 in forward

    def forward(self, z):
        # z shape expected (B, noice_dim)
        z = z.view(-1, self.noice_dim, 1, 1)
        out = self.net(z)               # -> (B,1,32,32)
        # center crop to 28x28
        _, _, h, w = out.shape
        top = (h - 28) // 2
        left = (w - 28) // 2
        out = out[:, :, top:top+28, left:left+28]
        return out
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        # batchnorm weight to N(1, 0.02), bias to 0
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
        


def discriminator_loss(logits_real, logits_fake, smoothing=0.9, noisy_labels=0.05):
    """
    One-sided label smoothing: real labels are smoothed to 0.9.
    Add small label noise optionally.
    """
    batch_size = logits_real.size(0)
    # smoothed real labels
    real_labels = torch.full_like(logits_real, smoothing)
    fake_labels = torch.zeros_like(logits_fake)

    # add small random noise to labels to regularize D (label flipping/noise)
    if noisy_labels > 0.0:
        real_labels = real_labels + (torch.rand_like(real_labels) - 0.5) * noisy_labels
        fake_labels = fake_labels + (torch.rand_like(fake_labels) - 0.5) * noisy_labels
        
    bce_loss = nn.BCEWithLogitsLoss()
    loss_real = bce_loss(logits_real, real_labels)
    loss_fake = bce_loss(logits_fake, fake_labels)
    return (loss_real + loss_fake) * 0.5

def generator_loss(logits_fake):
    # Generator tries to make D predict real (1)
    real_labels = torch.ones_like(logits_fake)
    bce_loss = nn.BCEWithLogitsLoss()
    return bce_loss(logits_fake, real_labels)
