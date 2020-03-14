import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from core.models.norm import CategoricalConditionalBatchNorm

def get_normalize_layer(layer, normalize, out_ch):
    if normalize not in ['batch', 'conditional_batch', 'instance', 'spectral', 'spectral+batch', 'spectral+instance', 'conditional_instance', None]:
        raise ValueError(f'{normalize} is invalid value for argument normalize')

    if normalize and normalize.startswith('spectral'):
        layer = spectral_norm(layer)
            
    # Normalization layer
    norm_layer = None
    if normalize in ['batch', 'spectral+batch']:
        norm_layer = nn.BatchNorm2d(out_ch)
    elif normalize == 'conditional_batch':
        norm_layer = CategoricalConditionalBatchNorm(out_ch)
    elif normalize in ['instance', 'spectral+instance']:
        norm_layer = nn.InstanceNorm2d(out_ch)
    elif normalize == 'conditional_instance':
        norm_layer = nn.ConditionalInstanceNorm(out_ch)

    return layer, norm_layer


class conv_layer(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=(3,1), stride=(1,1), pad=(1,0), dilation=(1,1), groups=1, bias=True,  normalize=None, n_class=1, use_wscale=False, init_mean=0, init_std=0.02, padding_mode='zeros'):
        super(conv_layer, self).__init__()

        if padding_mode == 'reflect':
            if pad == (0,0):
                self.pad = None
            else:
                self.pad = torch.nn.ReflectionPad2d((pad[1],pad[1],pad[0],pad[0])) 
                pad = (0,0)
        elif padding_mode == 'zeros':
            self.pad = None
        else:
            raise ValueError('invaild padding mode.')

        conv = nn.Conv2d(in_ch, out_ch, ksize, stride, pad, dilation=dilation, groups=groups, bias=bias)
        self.conv, self.norm_layer = get_normalize_layer(conv, normalize, out_ch)

        nn.init.normal_(self.conv.weight, mean=init_mean, std=init_std)
        if bias:
            nn.init.zeros_(self.conv.bias)

        self.n_class = n_class

    def forward(self, x, y=None):
        h = x
        if self.pad is not None:
            h = self.pad(h)
        h = self.conv(h)
        if self.norm_layer is not None:
            h = self.norm_layer(h)
        return h



class deconv_layer(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=(3,1), stride=(1,1), pad=(1,0), dilation=(1,1), groups=1, bias=True, normalize=None, n_class=1, use_wscale=False, padding_mode='zeros', init_mean=0, init_std=0.02):
        super(deconv_layer, self).__init__()

        self.ksize = ksize
        self.pad_size = pad
        self.stride = stride

        if padding_mode == 'reflect':
            if pad == (0,0):
                self.pad = None
            else:
                self.pad = torch.nn.ReflectionPad2d((0,0,0,0)) 
                pad = (0,0)
        elif padding_mode == 'zeros':
            self.pad = None
        else:
            raise ValueError('invaild padding mode.')

        deconv = nn.ConvTranspose2d(in_ch, out_ch, ksize, stride, pad, dilation=dilation, groups=groups, bias=bias)
        self.deconv, self.norm_layer = get_normalize_layer(deconv, normalize, out_ch)

        nn.init.normal_(self.deconv.weight, mean=init_mean, std=init_std)
        if bias:
            nn.init.zeros_(self.deconv.bias)

        self.n_class = n_class

    def forward(self, x, y=None):
        h = x
        if self.pad is not None:
             h = self.pad(h)
        h = self.deconv(h)
        if self.norm_layer is not None:
            h = self.norm_layer(h)

        if self.pad is not None:
             if self.stride[0] == 1:
                 h = h[:,:,self.ksize[0]//2:h.shape[2]-self.ksize[0]//2,:]
             if self.stride[1] == 1:
                 h = h[:,:,:,self.ksize[1]//2:h.shape[2]-self.ksize[1]//2]
        return h


class conv1d_layer(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, pad=1, dilation=1, groups=1, bias=True,  normalize=None, n_class=1, use_wscale=False, init_mean=0, init_std=0.01, padding_mode='zero'):
        super(conv1d_layer, self).__init__()

        if padding_mode == 'reflect':
            if ksize==2:
                self.pad = torch.nn.ReflectionPad1d((int((ksize-1)*dilation), 0))
            else:
                self.pad = torch.nn.ReflectionPad1d(pad) 
            pad = 0
        elif padding_mode == 'zeros':
            self.pad = None
        else:
            raise ValueError('invaild padding mode.')

        conv = nn.Conv1d(in_ch, out_ch, ksize, stride, pad, dilation=dilation, groups=groups, bias=bias)
        self.conv, self.norm_layer = get_normalize_layer(conv, normalize, out_ch)

        nn.init.normal_(self.conv.weight, mean=init_mean, std=init_std)
        if bias:
            nn.init.zeros_(self.conv.bias)

        self.n_class = n_class

    def forward(self, x, y=None):
        h = x
        if self.pad is not None:
            h = self.pad(h)
        h = self.conv(h)
        if self.norm_layer is not None:
            bs, ch, ts = h.shape
            h = h.view(bs, ch, ts, 1)
            h = self.norm_layer(h)
            h = h.view(bs, ch, ts)
            
        return h

