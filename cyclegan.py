import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from losses import SupConLoss, ContrastLoss, suContrastLoss, L1ContrastLoss

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *


import torch.nn as nn
import torch.nn.functional as F
import torch


FEEDBACK_ITER = 6

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="derain_cyclegan", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--lambda_contrast", type=float, default=0.00, help="contrastive loss weight")
parser.add_argument("--num_steps", type=int, default=4, help="time steps for convlstm")
opt = parser.parse_args()
print(opt)



def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.cpu().data.numpy() 
    img2 = img2.cpu().data.numpy()

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    print("MSE: {}".format(mse))
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)


# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("attention/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("background/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_alpha1 = torch.nn.MSELoss()
criterion_alpha2 = torch.nn.L1Loss()
criterion_pix_contrast = ContrastLoss()
criterion_contrast = L1ContrastLoss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.num_steps)  
D_B = Discriminator(input_shape)
D_A = Discriminator(input_shape)

out_shape = D_B.output_shape


if cuda:
    G_AB = nn.DataParallel(G_AB).cuda()
    D_B = nn.DataParallel(D_B).cuda()
    D_A = nn.DataParallel(D_A).cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))

else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    #D_B.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(G_AB.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(480), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  ## pixel value range: 0 to 1
]

transforms_2 = [
    transforms.Resize(int(480), Image.BICUBIC),
    transforms.CenterCrop((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset("/home/xulei/projects/fbnet/dataset/RainDS/RainDS_real_size/train_set", transforms_=transforms_, unaligned=False),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("/home/xulei/projects/fbnet/dataset/RainDS/RainDS_real_size/test_set", transforms_=transforms_2, unaligned=False),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)


#criterion_pix_contrast = ContrastLoss()

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    alpha = Variable(Tensor(np.zeros((real_A.size(0), 1, real_A.size(2), real_A.size(3)))), requires_grad=False)

    attention_maps = []
    background_maps = []

    for _ in range(FEEDBACK_ITER):
        B, R, alpha = G_AB( torch.cat((real_A, alpha), dim=1) )
        A_rec = (1 - alpha) * B + alpha * R
        attention_maps.append(alpha)
        background_maps.append(B)

    for i in range(real_B.size(0)):
        d = calc_psnr(B[i], real_B[i])
        print("PSNR {}: {}".format(i,d))

    real_A_att = real_A

    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=False)
    B = make_grid(B, nrow=5, normalize=False)
    A_rec = make_grid(A_rec, nrow=5, normalize=False)
    R = make_grid(R, nrow=5, normalize=False)
    alpha = make_grid(alpha, nrow=5, normalize=False)
    real_B = make_grid(real_B, nrow=5, normalize=False)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, B, R, alpha, A_rec, real_B), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)

    real_A = make_grid(real_A_att, nrow=5, normalize=False)
    for i in range(FEEDBACK_ITER):
        attention_maps[i] = make_grid(attention_maps[i], nrow=5, normalize=False)
    att_grid = torch.cat((real_A, *attention_maps), 1)
    save_image(att_grid, "attention/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)

    real_A = make_grid(real_A_att, nrow=5, normalize=False)
    for i in range(FEEDBACK_ITER):
        background_maps[i] = make_grid(background_maps[i], nrow=5, normalize=False)
    bg_grid = torch.cat((real_A, *background_maps), 1)
    save_image(bg_grid, "background/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)
    

# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *out_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *out_shape))), requires_grad=False)

        # used for feedback
        alpha = Variable(Tensor(np.zeros((real_A.size(0), 1, real_A.size(2), real_A.size(3)))), requires_grad=False)
        alpha_id = Variable(Tensor(np.zeros((real_A.size(0), 1, real_A.size(2), real_A.size(3)))), requires_grad=False)
        ones = Variable(Tensor(np.ones((real_A.size(0), 1, real_A.size(2), real_A.size(3)))), requires_grad=False)
        zeros = Variable(Tensor(np.zeros((real_A.size(0), 1, real_A.size(2), real_A.size(3)))), requires_grad=False)
        zeros_ch3 = Variable(Tensor(np.zeros((real_A.size()))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        optimizer_G.zero_grad()
        loss_G = 0
        mu = 2
        beta = 1.5

        for it in range(FEEDBACK_ITER):

            # Identity loss
            B_id, R_id, alpha_id= G_AB(torch.cat((real_B, alpha_id), dim=1))
            loss_id = criterion_identity(B_id, real_B)

            B, R, alpha = G_AB( torch.cat((real_A, alpha), dim=1) )

            DB_out = D_B(B)
            loss_GAN_B = criterion_GAN(DB_out, valid) 

            fake_A = (1 - alpha) * B + alpha * R

            DA_out = D_A(fake_A)
            loss_GAN_A = criterion_GAN(DA_out, valid)

            loss_GAN = (loss_GAN_A + loss_GAN_B) / 2.0
            loss_cycle = criterion_cycle(fake_A, real_A)

            hard_mask = torch.where(alpha<0.3, ones, zeros)
            alpha_loss_1 = 10 * criterion_alpha1(hard_mask * (real_A - B), zeros_ch3)
            alpha_loss_2 = 0.5 * criterion_alpha2(alpha, zeros)
            alpha_loss = (alpha_loss_1 + alpha_loss_2) / 2.0

            con_loss = criterion_contrast(B, real_B, real_A)#criterion_pix_contrast(B, real_B, real_A)


            loss_G += (opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_id + alpha_loss) + mu*loss_GAN + con_loss * opt.lambda_contrast
            mu *= beta

        
        loss_G.backward()
        optimizer_G.step()


        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        D_A_real = D_A(real_A)
        loss_real = criterion_GAN(D_A_real, valid)

        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        D_A_fake = D_A(fake_A_.detach())
        loss_fake = criterion_GAN(D_A_fake, fake)

        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()


        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        D_B_real = D_B(real_B)
        loss_real = criterion_GAN(D_B_real, valid)

        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(B)
        D_B_fake = D_B(fake_B_.detach())
        loss_fake = criterion_GAN(D_B_fake, fake)

        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2
        # --------------
        #  Log Progress
        # --------------

        # Determnie approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, id: %f, con_loss: %f] ETA: %s"
                #"\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, id: %f, a1: %f, a2: %f, con_loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_id.item(),
                #alpha_loss_1.item(),
                #alpha_loss_2.item(),
                con_loss.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            torch.no_grad()
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
