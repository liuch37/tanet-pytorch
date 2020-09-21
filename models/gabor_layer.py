'''
This module is to build Gabor CNN layer.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb

__all__ = ['gabor_layer']

def getGaborKernel(ksize, sigma, theta, lambd, gamma, psi):
    '''
    Compute Gabor kernel with PyTorch tensor - with numerical stability
    '''
    eps = 10**(-12)
    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    sigma_x = sigma
    sigma_y = sigma / (gamma + eps)
    nstds = 3
    c = torch.cos(theta)
    s = torch.sin(theta)

    if ksize > 0:
        xmax = ksize//2
    else:
        xmax = torch.round(torch.max(torch.abs(nstds*sigma_x*c), torch.abs(nstds*sigma_y*s)))

    if ksize > 0:
        ymax = ksize//2
    else:
        ymax = torch.round(torch.max(torch.abs(nstds*sigma_x*s), torch.abs(nstds*sigma_y*c)))

    xmin = -xmax
    ymin = -ymax

    kernel = torch.zeros((ymax - ymin + 1, xmax - xmin + 1))
    scale = 1
    ex = -0.5/(sigma_x*sigma_x+eps)
    ey = -0.5/(sigma_y*sigma_y+eps)
    cscale = torch.pi*2/(lambd + eps)

    for y in range(ymin, ymax+1):
        for x in range(xmin, xmax+1):
            xr = x*c + y*s
            yr = -x*s + y*c
            v = scale*torch.exp(ex*xr*xr + ey*yr*yr)*torch.cos(cscale*xr + psi)
            kernel[ymax - y, xmax - x] = v

    return kernel

def kernel_normalization(kernel):
    factor = torch.sqrt((kernel**2).sum())
    return kernel / factor

class gabor_layer(nn.Module):
    def __init__(self, out_planes, kernel_size=3):
        super(gabor_layer, self).__init__()
        self.padding = (kernel_size-1)//2
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.conv2d_h = F.conv2d
        self.conv2d_v = F.conv2d
        self.sigma = Parameter(torch.Tensor([5.0]))
        self.theta_h = torch.Tensor([0]) # no gradient
        self.theta_v = torch.Tensor([self.pi/2]) # no gradient
        self.lambd = Parameter(torch.Tensor([0.1]))
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.psi = Parameter(torch.Tensor([0.0]))
        self.kernel_horizontal = getGaborKernel(kernel_size, self.sigma, self.theta_h, self.lambd, self.gamma, self.psi)
        self.kernel_horizontal_normalized = kernel_normalization(self.kernel_horizontal)
        self.kernel_vertical = getGaborKernel(kernel_size, self.sigma, self.theta_v, self.lambd, self.gamma, self.psi)
        self.kernel_vertical_normalized = kernel_normalization(self.kernel_vertical)

        self.kernel_weight_horizontal = Parameter(self.kernel_horizontal_normalized.unsqueeze(0).unsqueeze(0).repeat(out_planes,1,1,1))
        self.kernel_weight_vertical = Parameter(self.kernel_vertical_normalized.unsqueeze(0).unsqueeze(0).repeat(out_planes,1,1,1))

    def forward(self, imgs):
        '''
        imgs should be in channel order R,G,B - size [batch, 3, H, W]
        gray = 0.299 R + 0.587 G + 0.114 B
        output imgs: size [batch, 1, H, W]
        '''
        imgs_gray = 0.299 * imgs[:,0:1,:,:] + 0.587 * imgs[:,1:2,:,:] + 0.114 * imgs[:,2:3,:,:]
        # Gabor convolution
        imgs_h = self.conv2d_h(imgs_gray, self.kernel_weight_horizontal, stride=1, padding=self.padding)
        imgs_v = self.conv2d_v(imgs_gray, self.kernel_weight_vertical, stride=1, padding=self.padding)

        return torch.sqrt(imgs_h**2 + imgs_v**2)

# unit testing
if __name__ == '__main__':
    # fake input image
    batch_size = 8
    Height = 32
    Width = 64
    Channel = 3
    with torch.autograd.set_detect_anomaly(True):
        input_images = torch.randn(batch_size,Channel,Height,Width)
        gabor = gabor_layer(1, kernel_size=3)
        print("Parameters before back propagation:", gabor.sigma, gabor.theta_h, gabor.theta_v, gabor.lambd, gabor.gamma, gabor.psi)
        input_filtered_images = gabor(input_images)
        print(input_filtered_images)
        print(input_filtered_images.shape)
        print(input_filtered_images.requires_grad)
        optimizer = optim.Adam(gabor.parameters(), lr=1.0)

        loss_total = torch.sum(input_filtered_images**2)
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        print("Parameters after back propagation:", gabor.sigma, gabor.theta_h, gabor.theta_v, gabor.lambd, gabor.gamma, gabor.psi)

    # real input image
    test_image_path = '../misc/1_1.png'
    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    imgs = torch.FloatTensor(img)
    imgs = imgs.permute(2,0,1)
    imgs = imgs.unsqueeze(0)
    imgs_filtered = gabor(imgs)
    imgs_filtered_numpy = imgs_filtered[0,0,:,:].detach().numpy()
    plt.imshow(imgs_filtered_numpy, cmap='gray')
    plt.show()