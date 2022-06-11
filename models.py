import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from spectral import SpectralNorm
#from DenseNet import _DenseBlock


def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.dense_blk = nn.Sequential(nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False), nn.ReLU(inplace=True))
    def forward(self, x):
        out = self.dense_blk(x)
        out = torch.cat((x, out), 1)
        return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_steps, growth_rate=32, layer=6, drop_rate=0, memory_efficient=False):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]
        in_features = channels + 1

        # Initial convolution block
        out_features = 64
        self.init_conv = nn.Sequential(nn.ReflectionPad2d(channels),
            nn.Conv2d(in_features, out_features, 7),
            nn.ReLU(inplace=True))
        in_features = out_features

        self.RDB1 = RDB(in_features, nDenselayer=layer, growthRate=growth_rate)
        self.RDB2 = RDB(in_features, nDenselayer=layer, growthRate=growth_rate)
        #self.RDB3 = RDB(in_features, nDenselayer=layer, growthRate=growth_rate)

        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(in_features*2, in_features, kernel_size=1, padding=0, bias=True)
      
        self.gen_out = nn.Sequential(nn.ReflectionPad2d(channels), nn.Conv2d(in_features, 7, 7))
        self.alpha_conv = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())
        self.BR_conv = nn.Sequential(nn.Conv2d(7, channels*2, 1), nn.Sigmoid())

    def forward(self, x):
        F_0 = self.init_conv(x)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        #F_3 = self.RDB3(F_2)     
        FF = torch.cat((F_1, F_2), 1)
        FdLF = self.GFF_1x1(FF)
        out = self.gen_out(FdLF)
        br , alpha = torch.split(out, [6, 1], dim=1)

        alpha = self.alpha_conv(alpha)
        BR_alpha = torch.cat((br,alpha), 1)
        BR = self.BR_conv(BR_alpha)
        b,r = torch.split(BR, [3,3], dim=1)
        return b, r, alpha



##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)


        def discriminator_block(in_filters, out_filters, normalize=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [SpectralNorm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            SpectralNorm(nn.Conv2d(512, 1, 4, padding=1))
        )


    def forward(self, img):
        return self.model(img)




