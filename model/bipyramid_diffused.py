import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
class HINBlock(nn.Module):
    #Half Instance Normalization Block
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(HINBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        initialize_weights_xavier([self.conv_1, self.conv_2], 0.1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out
class AFF(nn.Module):
    def __init__(self,channel,reduction):
        super(AFF,self).__init__()
        self.prefuse = nn.Conv2d(2*channel,channel,3,1,1)
        self.glob = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel,channel//reduction,1,1,0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel//reduction,channel,1,1,0))
        self.loca = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,1,0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel//reduction,channel,1,1,0))
        self.act= nn.Sigmoid()
    def forward(self,x,y):
        prem = self.prefuse(torch.cat([x,y],dim=1))
        atten = self.act(self.glob(prem)+self.loca(prem))
        return x*(1-atten)+y*(atten)
class FreAdjust(nn.Module):
    def __init__(self,channels):
        super(FreAdjust,self).__init__()
        self.sft = SFT(channels)
        self.id = nn.Conv2d(channels,channels,3,1,1)
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(2*channels,channels,1,1,0),
            nn.LeakyReLU(0.1,inplace=False),
            nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(2*channels,channels,1,1,0),
            nn.LeakyReLU(0.1,inplace=False),
            nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)

    def forward(self,msf,panf):
        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        pha_fuse = self.pha_fuse(torch.cat([msF_pha, panF_pha], 1))
        msF_amp = self.sft(msF_amp,panF_amp)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        return self.post(out)+self.id(msf)

class SFT(nn.Module):
    def __init__(self,nc):
        super(SFT,self).__init__()
        self.convmul = nn.Conv2d(nc,nc,3,1,1)
        self.convadd = nn.Conv2d(nc,nc,3,1,1)
        self.convfuse = nn.Conv2d(2*nc,nc,1,1,0)
    def forward(self,ms,pan):
        mul = self.convmul(pan)
        add = self.convadd(pan)
        return ms*mul+add
class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(2*channels,channels,1,1,0),
            nn.LeakyReLU(0.1,inplace=False),
            nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(2*channels,channels,1,1,0),
            nn.LeakyReLU(0.1,inplace=False),
            nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)

def downsample(x,h,w):
    pass
def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

class FeaturExtractionPyramid(nn.Module):
    def __init__(self,num_channels):
        super(FeaturExtractionPyramid, self).__init__()
        self.down1 = nn.Conv2d(num_channels, 2*num_channels, 3, 2, 1)
        self.down2 = nn.Conv2d(2*num_channels, 4*num_channels, 3, 2, 1)
        self.conv1 = HINBlock(num_channels, num_channels)
        self.conv2 = HINBlock(num_channels, num_channels)
        self.conv3 = HINBlock(num_channels, num_channels)
        self.conv4 = HINBlock(num_channels, num_channels)
        self.conv5 = HINBlock(2*num_channels, 2*num_channels)
        self.conv6 = HINBlock(2*num_channels, 2*num_channels)
    def forward(self,panf):
        p1,p2,p3 = None,None,None
        panf = self.conv1(panf)
        panf = self.conv2(panf)
        p1 = panf
        panf = self.conv3(panf)
        panf = self.conv4(panf)
        panf = self.down1(panf)
        p2 = panf
        panf = self.conv5(panf)
        panf = self.conv6(panf)
        panf = self.down2(panf)
        p3 = panf
        return p1,p2,p3

class FeaturExtractionPyramid_diffused(nn.Module):
    def __init__(self,num_channels):
        super(FeaturExtractionPyramid_diffused, self).__init__()
        self.down1 = nn.Conv2d(num_channels, 2*num_channels, 3, 2, 1)
        self.down2 = nn.Conv2d(2*num_channels, 4*num_channels, 3, 2, 1)
        self.conv1 = HINBlock(num_channels, num_channels)
        self.conv2 = HINBlock(num_channels, num_channels)
        self.conv3 = HINBlock(num_channels, num_channels)
        self.conv4 = HINBlock(num_channels, num_channels)
        self.conv5 = HINBlock(2*num_channels, 2*num_channels)
        self.conv6 = HINBlock(2*num_channels, 2*num_channels)
        self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))

    def diffuse_func(self, img, guide, l=0.24, K=0.01, eps=1e-8, ):
        # _, _, h, w = guide.shape

        # Convert the features to coefficients with the Perona-Malik edge-detection function
        cv, ch = c(guide, K=K)

        img = diffuse_step(cv, ch, img, l=l)

        return img
    def forward(self,panf):
        p1,p2,p3 = None,None,None
        panf = self.conv1(panf)
        panf = self.conv2(panf)
        p1 = self.diffuse_func(panf, panf, K=torch.exp(self.logk))
        panf = self.conv3(panf)
        panf = self.conv4(panf)
        panf = self.down1(panf)
        p2 = self.diffuse_func(panf, panf, K=torch.exp(self.logk))
        panf = self.conv5(panf)
        panf = self.conv6(panf)
        panf = self.down2(panf)
        p3 = self.diffuse_func(panf, panf, K=torch.exp(self.logk))
        return p1,p2,p3

class SpatialInjectionPyramid(nn.Module):
    def __init__(self,num_channels):
        super(SpatialInjectionPyramid,self).__init__()
        channels = num_channels
        self.SIM1 = AFF(channels,16)
        self.SIM2 = AFF(2*channels, 16)
        self.SIM3 = AFF(4*channels,16)
        self.SIM4 = AFF(2*channels, 16)
        self.SIM5 = AFF(channels,16)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(2*channels, 2*channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(4*channels, 4*channels, 3, 1, 1)
        self.down1 = nn.Conv2d(channels,2*channels,3,2,1)
        self.down2 = nn.Conv2d(2*channels, 4*channels, 3, 2, 1)
        self.conv4 = HINBlock(4*channels,2*channels)
        self.conv5 = HINBlock(2*channels,channels)
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
    def forward(self,msf,p1,p2,p3):
        msf = self.conv1(msf)
        msf = self.SIM1(msf,p1)
        u0 = msf
        msf = self.down1(msf)

        msf = self.conv2(msf)
        msf = self.SIM2(msf,p2)
        u1 = msf
        msf = self.down2(msf)

        msf = self.conv3(msf)
        msf = self.SIM3(msf,p3)
        m3 = msf
        msf = self.up1(torch.cat([msf,m3],dim=1))

        msf = self.SIM4(msf,p2)
        msf = self.up2(torch.cat([msf,u1],dim=1))
        msf = self.SIM5(msf,p1)
        return msf
class FeatureInjectionModule(nn.Module):
    def __init__(self,num_channels):
        super(FeatureInjectionModule,self).__init__()
        channels = num_channels
        self.fft = Freprocess(channels)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
    def forward(self,msf,panf):
        fre = self.fft(msf,panf)
        spa = self.fuse(torch.cat([fre,msf],1))
        fuse = spa+msf
        return fuse
class FrequencyInjectionPyramid(nn.Module):
    def __init__(self,num_channels):
        super(FrequencyInjectionPyramid,self).__init__()
        channels = num_channels
        self.conv1 = HINBlock(channels, channels)
        self.conv2 = HINBlock(2 * channels, 2 * channels)
        self.conv3 = nn.Conv2d(4 * channels, 4 * channels, 3, 1, 1)
        self.down1 = nn.Conv2d(channels, 2 * channels, 3, 2, 1)
        self.down2 = nn.Conv2d(2 * channels, 4 * channels, 3, 2, 1)
        self.conv4 = HINBlock(4 * channels, 2 * channels)
        self.conv5 = HINBlock(2 * channels, channels)
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
        self.FIM1 = FreAdjust(channels)
        self.FIM2 = FreAdjust(2*channels)
        self.FIM3 = FreAdjust(4*channels)
        self.FIM4 = FreAdjust(2*channels)
        self.FIM5 =FreAdjust(channels)
    def forward(self,msf,m1,m2,m3):
        _, _, M, N = msf.shape
        #msf channel
        #m1 channel
        #m2 2*channel
        #m3 4*channel
        msf = self.conv1(msf)
        msf = self.FIM1(msf,m1)
        u0 = msf
        msf = self.down1(msf)
        msf = self.conv2(msf)
        msf = self.FIM2(msf,m2)
        u1 = msf
        msf = self.down2(msf)
        msf = self.conv3(msf)
        msf = self.FIM3(msf,m3)
        u3 = msf
        msf = self.up1(torch.cat([msf,u3],dim=1))
        msf = self.FIM4(msf,m2)
        # msf = self.conv4(torch.cat([msf,u1],dim=1))
        msf = self.up2(torch.cat([msf,u1],dim=1))
        # msf = self.conv5(torch.cat([msf,u0],dim=1))
        msf = self.FIM5(msf,m1)
        return msf

class FrequencySpatialFusionModule(nn.Module):
    def __init__(self,num_channels):
        super(FrequencySpatialFusionModule,self).__init__()
        channels = num_channels
        self.conv1 = HINBlock(channels, 4 * channels)
        self.FIM1 = FreAdjust(4 * channels)
        self.SIM1 = AFF(4 * channels, 16)
        self.up1 = nn.PixelShuffle(2)
        self.conv2 = HINBlock(2 * channels, 2 * channels)
        self.FIM2 = FreAdjust(2 * channels)
        self.SIM2 = AFF(2 * channels, 16)
        self.up2 = nn.PixelShuffle(2)
        self.conv3 = HINBlock(channels, channels)
        self.FIM3 = FreAdjust(channels)
        self.SIM3 = AFF(channels, 16)
        self.conv4 = nn.Conv2d(2 * channels, channels, 1, 1, 0)

    def forward(self, msf, panf1, panf2, panf3):
        msf = self.conv1(msf)
        F_squ = self.FIM1(msf, panf3)
        S_squ = self.SIM1(msf, panf3)
        midout = self.up1(torch.cat([F_squ, S_squ], dim=1))
        msf = self.conv2(midout)
        F_squ = self.FIM2(msf, panf2)
        S_squ = self.SIM2(msf, panf2)
        midout = self.up2(torch.cat([F_squ, S_squ], dim=1))
        msf = self.conv3(midout)
        F_squ = self.FIM3(msf, panf1)
        S_squ = self.SIM3(msf, panf1)
        out = self.conv4(torch.cat([F_squ, S_squ], dim=1))

        return out

class Net(nn.Module):
    def __init__(self, num_channels=4,channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        channels = base_filter
        self.feature_extraction = FeaturExtractionPyramid_diffused(base_filter)
        '''
        self.spatialinjection = SpatialInjectionPyramid(base_filter)
        self.freinjection = FrequencyInjectionPyramid(base_filter)
        self.conv_ms = nn.Conv2d(4,channels,3,1,1)
        self.conv_pan = nn.Conv2d(1,channels,3,1,1)
        self.refine = Refine(channels,4)
        self.out = nn.Conv2d(channels,4,3,1,1)
        '''
        self.fsinjection = FrequencySpatialFusionModule(base_filter)
        self.conv_ms = nn.Conv2d(num_channels, channels,3,1,1)
        self.conv_pan = nn.Conv2d(1,channels,3,1,1)
        self.refine = Refine(channels,num_channels)

    def forward(self, ms,_,pan):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
         # size 4
        msf = self.conv_ms(ms)
        panf = self.conv_pan(pan)
        p1,p2,p3 = self.feature_extraction(panf)
        '''
        msf = self.freinjection(msf,p1,p2,p3)
        midout = self.out(msf)
        msf = self.spatialinjection(msf,p1,p2,p3)
        '''
        msf = self.fsinjection(msf, p1, p2, p3)
        mHR = upsample(ms, M, N)
        ms = mHR

        return self.refine(msf)+ms,msf

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3

class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out

@torch.jit.script
def g(x, K: float=0.03):
    # Perona-Malik edge detection
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))

@torch.jit.script
def c(I, K: float=0.03):
    # apply function to both dimensions
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]), 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]), 1), 1), K)
    return cv, ch

@torch.jit.script
def diffuse_step(cv, ch, I, l: float=0.24):
    # Anisotropic Diffusion implmentation, Eq. (1) in paper.

    # calculate diffusion update as increments
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv 

    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th 
    return I
