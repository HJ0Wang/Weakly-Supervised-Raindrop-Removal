
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=190, help="epoch for testing")
parser.add_argument("--dataset_name", type=str, default="derain_cyclegan", help="name of the dataset")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--patch_size", type=int, default=256, help="size of patch")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--num_steps", type=int, default=4, help="time steps for convlstm")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images_comb/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = torch.cuda.is_available()
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.num_steps)  ## apply attention for the derain process
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

out_shape = D_A.output_shape

if cuda:
    G_AB = nn.DataParallel(G_AB).cuda()
    D_A = nn.DataParallel(D_A).cuda()
    D_B = nn.DataParallel(D_B).cuda()


# Load pretrained models
G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


transforms_2 = [
    transforms.Resize(int(480), Image.BICUBIC),
    #transforms.CenterCrop((opt.img_height, opt.img_width)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


# Test data loader            #"/home/xulei/projects/fbnet/dataset/raindrop_data/test_a" 
val_dataloader = DataLoader(  #"/home/xulei/projects/fbnet/dataset/RainDS/RainDS_real_size/test_set"
    ImageDataset("/home/xulei/projects/fbnet/dataset/RainDS/RainDS_real_size/test_set", transforms_=transforms_2, unaligned=False),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

def get_all_patch_old(rainy, opt):
    patch_size = opt.patch_size
    ih, iw = rainy.shape[:2]
    num_height = math.floor(ih / patch_size)
    num_width = math.floor(iw / patch_size)

    if ih % patch_size != 0:
    	num_height += 1
    if iw % patch_size != 0:
    	num_width += 1

    num_total = num_height * num_width

    tmp = np.zeros([num_height * patch_size, num_width * patch_size, 3 ], dtype=float)
    tmp[ :ih, :iw, :] = rainy 

    # print(lr.shape,tmp.shape)
    X = np.zeros([num_total, patch_size, patch_size, 3], dtype=float)

    cnt = 0

    for x_axis in range(num_width):
        for y_axis in range(num_height):
            ix = x_axis * patch_size
            iy = y_axis * patch_size

            X[cnt] = tmp[iy:iy + patch_size, ix:ix + patch_size, :]
            cnt += 1
    X = np.array(X)

    # print(num_total)
    return X, num_height, num_width


def merge_image_old(frames, num_height, num_width, opt):

    patch_size = opt.patch_size
    
    merged_frame = np.zeros([patch_size * num_height, patch_size * num_width, 3], dtype=float)
    # ip = patch_size
    cnt = 0

    for x_axis in range(num_width):
        for y_axis in range(num_height):
            ix = x_axis * patch_size
            iy = y_axis * patch_size

            merged_frame[iy:iy + patch_size, ix:ix + patch_size, :] = frames[cnt]
            cnt += 1

    return merged_frame


def get_all_patch(rainy, opt):
    patch_size = opt.patch_size
    ih, iw = rainy.shape[:2]
    num_height = math.floor(ih / patch_size)
    num_width = math.floor(iw / patch_size)

    if ih % patch_size != 0:
        num_height += 1
    if iw % patch_size != 0:
        num_width += 1

    num_total = num_height * num_width

    tmp = np.zeros([num_height * patch_size, num_width * patch_size, 3 ], dtype=float)
    tmp[ :ih, :iw, :] = rainy

    for nh in range(num_height-1):
        tmp[nh*patch_size:(nh+1)*patch_size, -patch_size:, :] = rainy[nh*patch_size:(nh+1)*patch_size, -patch_size:, :]
    for nw in range(num_width-1):
        tmp[-patch_size:, nw*patch_size:(nw+1)*patch_size:, :] = rainy[-patch_size:, nw*patch_size:(nw+1)*patch_size, :]

    tmp[-patch_size:, -patch_size:, :] = rainy[-patch_size:, -patch_size:, :]


    # print(lr.shape,tmp.shape)
    X = np.zeros([num_total, patch_size, patch_size, 3], dtype=float)

    cnt = 0

    for x_axis in range(num_width):
        for y_axis in range(num_height):
            ix = x_axis * patch_size
            iy = y_axis * patch_size

            X[cnt] = tmp[iy:iy + patch_size, ix:ix + patch_size, :]
            cnt += 1
    X = np.array(X)

    # print(num_total)
    return X, num_height, num_width

def merge_image(frames, ih, iw, num_height, num_width, opt):

    patch_size = opt.patch_size
    
    merged_frame = np.zeros([patch_size * num_height, patch_size * num_width, 3], dtype=float)
    # ip = patch_size
    cnt = 0

    for x_axis in range(num_width):
        for y_axis in range(num_height):
            ix = x_axis * patch_size
            iy = y_axis * patch_size

            merged_frame[iy:iy + patch_size, ix:ix + patch_size, :] = frames[cnt]
            cnt += 1

    new_frame = merged_frame
    for nh in range(num_height-1):
        new_frame[nh*patch_size:(nh+1)*patch_size, iw-patch_size:iw, :] = merged_frame[nh*patch_size:(nh+1)*patch_size, -patch_size:, :]
    for nw in range(num_width-1):
        new_frame[ih-patch_size:ih, nw*patch_size:(nw+1)*patch_size, :] = merged_frame[-patch_size:, nw*patch_size:(nw+1)*patch_size, :]
    new_frame[(ih-patch_size):ih, (iw-patch_size):iw, :] = merged_frame[-patch_size:, -patch_size:, :]

    return new_frame


def calc_psnr_v3(img1, img2):
    #im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    img1 = img1.transpose(1, 2, 0)
    img2 = img2.transpose(1, 2, 0)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return  peak_signal_noise_ratio(img1, img2, data_range=1.0)
    #return peak_signal_noise_ratio(im1, im2, data_range=1.0)


def calc_ssim_norm(img1, img2):
    #im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #img1 = img1.transpose(1, 2, 0)
    #img2 = img2.transpose(1, 2, 0)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    alpha = np.sum(img2) / np.sum(img1)
    img1 *= alpha
    return structural_similarity(img1, img2, multichannel=True, data_range=1.0)


def calc_ssim(img1, img2):
    #im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #img1 = img1.transpose(1, 2, 0)
    #img2 = img2.transpose(1, 2, 0)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return structural_similarity(img1, img2, multichannel=True, data_range=1.0)




def calc_psnr_v2(img1, img2):
    img1 = img1.transpose(1, 2, 0)
    img2 = img2.transpose(1, 2, 0)
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have same shape!")
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))


def calc_psnr_v1_norm(img1, img2):

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    alpha = np.sum(img2) / np.sum(img1)
    img1 *= alpha
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)


def calc_psnr_v1(img1, img2):
    # img1 and img2 have range [0, 1]
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)

def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    min1 = np.amin(img1)
    min2 = np.amin(img2)

    img1 -= min1
    img2 -= min2

    max1 = np.amax(img1)
    max2 = np.amax(img2)

    img1 /= max1
    img2 /= max2

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)


def sample_images(opt):
    """Saves a generated sample from the test set"""
    ps = []
    ps1 = [] 
    ps2 = []
    ps3 = []
    ps1n = []
    ss = []
    ssn = []
    for ind,batch in enumerate(val_dataloader):
        img_A = batch["A"][0].cpu().data.numpy()
        img_B = batch["B"][0].cpu().data.numpy()
        img_A = np.transpose(img_A, (1,2,0))
        img_B = np.transpose(img_B, (1,2,0))
        #print("shape of img_A: {}".format(img_A.shape))

        ih, iw = img_A.shape[:2]

        frames_A, nH, nW = get_all_patch(img_A, opt)

        G_AB.eval()
        frames = []

        for i in range(nW * nH):
            inp = [np.transpose(frames_A[i], (2,0,1))]
            real_A = Variable(Tensor(inp))
            alpha = Variable(Tensor(np.zeros((real_A.size(0), 1, real_A.size(2), real_A.size(3)))), requires_grad=False)

            for i in range(6):
                B, R, alpha = G_AB(torch.cat((real_A, alpha), dim=1))

            B = np.transpose(B.cpu().data.numpy()[0], (1,2,0))

            frames.append(B)

        merged_frame = merge_image(frames, ih, iw, nH, nW, opt)
        merged_frame = merged_frame[:ih, :iw, :]
        psnr = calc_psnr(merged_frame, img_B)
        psnr1 = calc_psnr_v1(merged_frame, img_B)
        psnr2 = calc_psnr_v2(merged_frame, img_B)
        psnr3 =  calc_psnr_v3(merged_frame, img_B)
        psnr1n =  calc_psnr_v1_norm(merged_frame, img_B)
        ssim = calc_ssim(merged_frame, img_B)
        ssimn = calc_ssim_norm(merged_frame, img_B)
        print(ind, psnr, psnr1, psnr2, psnr3, psnr1n, ssim, ssimn)

        merged_frame = Variable(Tensor(np.transpose(merged_frame, (2,0,1))))

        save_image(merged_frame, "images_comb/%s/%s_fake.png" % (opt.dataset_name, ind), normalize=True)
        save_image(batch["B"][0], "images_comb/%s/%s_real.png" % (opt.dataset_name, ind), normalize=True)
        ps.append(psnr)
        ss.append(ssim)
        ps1.append(psnr1)
        ps2.append(psnr2)
        ps3.append(psnr3)
        ps1n.append(psnr1n)
        ssn.append(ssimn)
    print("AVG PSNR: {} {} {} {} {}".format(sum(ps)/len(ps), sum(ps1)/len(ps1),sum(ps2)/len(ps2),sum(ps3)/len(ps3),sum(ps1n)/len(ps1n)))
    print("AVG SSIM: {} {}".format(sum(ss)/len(ss), sum(ssn)/len(ssn)))

def main():
    sample_images(opt)



if __name__ == '__main__':
	main()


