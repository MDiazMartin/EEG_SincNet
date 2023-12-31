import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math


torch.manual_seed(2022)
np.random.seed(2022)
# random.seed(2021)


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    # x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
    #                   -1, -1), ('cpu')[x])().long(), :]   # flip left and right
    # x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,-1, -1), 'cpu').long(), :]
    x = x.view(x.size(0), x.size(1), -1)[:, torch.flip(torch.arange(x.size(1)-1, -1, -1).long(), dims=[0]), :]


    return x.view(xsize)

def sinc(band,t_right):

    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)
    y=torch.cat([y_left,Variable(torch.ones(1)),y_right])

    return y


class sinc_conv(nn.Module):
    '''
    this function is directly using the methods from Ravanelli et al. https://github.com/mravanelli/SincNet
    the trainable parameters are technically f1 and fband
    '''

    def __init__(self, N_filt, Filt_dim, fs, cutoff):
        super(sinc_conv, self).__init__()


        low_freq = 0
        high_freq= 200
        freq_init = np.random.uniform(low_freq,high_freq, N_filt)


        b1 = freq_init
        b2 = np.zeros_like(b1) + 2

        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy(b2 / self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.cutoff =  cutoff
    def forward(self, x):
        filters = Variable(torch.zeros((self.N_filt, self.Filt_dim)))
        N = self.Filt_dim
        t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs)
        min_freq = 1.0
        min_band = 1.0
        # filt_beg_freq = torch.clamp(torch.abs(self.filt_b1) +  min_freq / self.freq_scale, min_freq / self.freq_scale, \
        #                             (self.cutoff) / self.freq_scale - min_band/self.freq_scale)
        # filt_end_freq = torch.clamp(filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale), \
        #                             int(min_freq+min_band)/self.freq_scale, (self.cutoff)/self.freq_scale)
        # filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale

        filt_beg_freq =  torch.clamp(torch.abs(self.filt_b1) + min_freq / self.freq_scale, min_freq / self.freq_scale, \
                                     ((self.cutoff) - int(min_band)) / self.freq_scale)

        filt_end_freq = torch.clamp(filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale), \
                                    int(min_freq + min_band) / self.freq_scale, (self.cutoff) / self.freq_scale)

        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N)
        window = Variable(window.float())

        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i].float() * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)

            band_pass = band_pass / torch.max(band_pass)   #normalize to one

            filters[i, :] = band_pass * window
        batch_n = x.shape[0]
        x = x
        # return filters
        out = F.conv2d(x.view(batch_n,1,x.shape[1],x.shape[-1]), filters.view(self.N_filt, 1, 1,self.Filt_dim))

        return out



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, groups=in_channels, bias=bias)
        # self.pointwise = nn.Conv2d(out_channels*depth, out_channels*depth, kernel_size=[1,1], bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        # out = self.pointwise(out)
        return out


class SeparableConv2d_pointwise(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, bias=False):
        super(SeparableConv2d_pointwise, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(out_channels*depth, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

