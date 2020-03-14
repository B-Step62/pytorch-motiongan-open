import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.layers import conv_layer, deconv_layer

class GlobalResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_joint=27, g_div=8, normalize=None, activation='leaky_relu', ksize=(3,1), stride=(1,1), pad=(1,0), upsample=False, downsample=False, residual=True):
        super(GlobalResBlock, self).__init__()
        self.c_l = conv_layer(in_ch, in_ch, ksize, stride, pad, normalize=normalize)
        self.c_g = conv_layer(in_ch, n_joint*in_ch//g_div, ksize=(ksize[0], n_joint), stride=(stride[0],1), pad=(pad[0],0), normalize=normalize)
        self.c_c = conv_layer(in_ch+in_ch//g_div, out_ch, ksize, stride, pad, normalize=normalize)

        self.learnable_sc = in_ch != out_ch and residual
        if self.learnable_sc:
            self.c_sc = conv_layer(in_ch, out_ch, ksize=(1,1), stride=(1,1), pad=(0,0), normalize=normalize)

        #activation layer
        assert activation in ['leaky_relu', 'relu']
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
  
        self.upsample = upsample
        self.downsample = downsample 
        self.residual = residual
        self.n_joint = n_joint

    def __call__(self, x, y=None, outsize=None):
        bs, ch, t, wi = x.shape
        # source
        h = x
        if self.upsample:
            h = F.upsample(h, scale_factor=(2,1))
        h_l = self.activation(self.c_l(h, y=y))
        h_g = self.activation(self.c_g(h, y=y)).view(bs, -1, h_l.shape[2], self.n_joint)
        h = self.activation(self.c_c(torch.cat((h_l, h_g), dim=1), y=y))
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=(2,1))

        # residual
        if self.residual:
            res = self.c_sc(x, y=y) if self.learnable_sc else x
            if self.upsample:
                res = F.upsample(res, scale_factor=(2,1))
            if self.downsample:
                res = F.avg_pool2d(h, kernel_size=(2,1))
        return h + res if self.residual else h
