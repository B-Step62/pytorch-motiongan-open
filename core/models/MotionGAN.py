import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm

from core.models.layers import conv_layer, deconv_layer
from core.models.norm import PixelNormalizationLayer, AdaIN


#=================================================================================
#
###   Generator 
#
#=================================================================================

class MotionGAN_generator(nn.Module):
    def __init__(self, cfg, num_class):
        super(MotionGAN_generator, self).__init__()

        # Parameters for model initialization
        top = cfg.top
        padding_mode = cfg.padding_mode
        kw = cfg.kw
        w_dim = cfg.w_dim

        # Other settings
        self.cfg = cfg
        self.num_class = num_class
        self.z_dim = cfg.z_dim
        self.activation = nn.LeakyReLU()
        self.n_rows = 81
        self.num_adain = 14


        self.ec0_0 = conv_layer(1, top, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec1_0 = conv_layer(top, top*2, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec1_1 = conv_layer(top*2, top*2, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec2_0 = conv_layer(top*2, top*4, ksize=(kw,1), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec2_1 = conv_layer(top*4, top*4, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec3_0 = conv_layer(top*4, top*8, ksize=(kw,1), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec3_1 = conv_layer(top*8, top*8, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)

        self.dc_bottom = deconv_layer(top*8, top*8, ksize=(kw,3), stride=(1,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 

        self.dc3_1 = deconv_layer(top*8, top*8, ksize=(kw,3), stride=(1,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc3_0 = conv_layer(top*8, top*4, ksize=(kw,3), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc2_1 = deconv_layer(top*8, top*4, ksize=(kw, 3), stride=(1,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc2_0 = conv_layer(top*4, top*2, ksize=(kw,3), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc1_1 = deconv_layer(top*4, top*2, ksize=(kw, 3), stride=(1,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc1_0 = conv_layer(top*2, top, ksize=(kw,3), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc0_1 = deconv_layer(top, top, ksize=(kw, 3), stride=(1,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.dc0_0 = conv_layer(top, 1, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.c_traj = conv_layer(top, 3, ksize=(kw,self.n_rows), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)

        # style generator
        self.latent_transform = LatentTransformation(cfg, num_class)

        self.adain_dc_bototm = AdaIN(top*8, w_dim)
        self.adain_dc3_1 = AdaIN(top*8, w_dim)
        self.adain_dc3_0 = AdaIN(top*4, w_dim)
        self.adain_dc2_1 = AdaIN(top*4, w_dim)
        self.adain_dc2_0 = AdaIN(top*2, w_dim)
        self.adain_dc1_1 = AdaIN(top*2, w_dim)
        self.adain_dc1_0 = AdaIN(top, w_dim)

        self.adain_ec3_1 = AdaIN(top*8, w_dim)
        self.adain_ec3_0 = AdaIN(top*8, w_dim)
        self.adain_ec2_1 = AdaIN(top*4, w_dim)
        self.adain_ec2_0 = AdaIN(top*4, w_dim)
        self.adain_ec1_1 = AdaIN(top*2, w_dim)
        self.adain_ec1_0 = AdaIN(top*2, w_dim)
        self.adain_ec0_0 = AdaIN(top, w_dim)
        self.num_adain = 14
            

    # Generate noise
    def make_hidden(self, size, frame_nums, y=None):
        mode = self.cfg.use_z
        if not mode:
            z = None
        elif mode == 'transform':
            z = torch.randn(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor)
        elif mode == 'transform_const':
            z = torch.zeros(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor)
        elif mode == 'transform_uniform':
            z = torch.rand(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor) * 2 - 1.0
        else:
            raise ValueError(f'Invalid noise mode \"{mode}\"!!')
        return z

    def inference_adain(self, z, labels, layer_name):
        w = self.latent_transform(z, labels)
        adain_layer = getattr(self, layer_name)
        adain_params = adain_layer.eval_params(w) 
        return adain_params 


    def forward(self, control, z=None, labels=None, w=None):
        bs, _, ts, _ = control.shape 
        h = control 

        # transform z
        if w is None:
            w = self.latent_transform(z, labels)
        # duplicate w to input each AdaIN layer
        w = w.view(bs, 1, -1, 1, 1).expand(-1, self.num_adain, -1, -1, -1)

        ## Encoder
        e0 = self.activation(self.ec0_0(h))
        e0 = self.adain_ec0_0(e0, w[:,13])

        e1 = self.activation(self.ec1_0(e0))
        e1 = self.adain_ec1_0(e1, w[:,12])
        e1 = self.activation(self.ec1_1(e1))
        e1 = self.adain_ec1_1(e1, w[:,11])

        e2 = self.activation(self.ec2_0(e1))
        e2 = self.adain_ec2_0(e2, w[:,10])
        e2 = self.activation(self.ec2_1(e2))
        e2 = self.adain_ec2_1(e2, w[:,9])

        e3 = self.activation(self.ec3_0(e2))
        e3 = self.adain_ec3_0(e3, w[:,8])
        e3 = self.activation(self.ec3_1(e3))
        e3 = self.adain_ec3_1(e3, w[:,7])

        e3 = self.activation(self.dc_bottom(e3))
        e3 = self.adain_dc_bototm(e3, w[:,6])

        ## Decoder
        d2 = self.activation(self.dc3_1(F.interpolate(e3, scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d2 = self.adain_dc3_1(d2, w[:,5])
        d2 = self.activation(self.dc3_0(d2))
        d2 = self.adain_dc3_0(d2, w[:,4])

        d1 = self.activation(self.dc2_1(F.interpolate(torch.cat((e2.repeat(1,1,1,9), d2), dim=1), scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d1 = self.adain_dc2_1(d1, w[:,3])
        d1 = self.activation(self.dc2_0(d1))
        d1 = self.adain_dc2_0(d1, w[:,2])

        d0 = self.activation(self.dc1_1(F.interpolate(torch.cat((e1.repeat(1,1,1,27), d1), dim=1), scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d0 = self.adain_dc1_1(d0, w[:,1])
        d0 = self.activation(self.dc1_0(d0))
        d0 = self.adain_dc1_0(d0, w[:,0])

        ## Get trajectory
        traj = self.c_traj(d0)
        traj = traj.transpose(1,3)
        
        ## Get motion 
        motion = self.dc0_0(d0)
 
        return traj, motion 



#=================================================================================
#
###    Discriminator
#
#=================================================================================

class MotionGAN_discriminator(nn.Module):
    def __init__(self, cfg, frame_nums, num_class):
        super(MotionGAN_discriminator, self).__init__()

        # Parameters for model initialization
        top = cfg.top
        padding_mode = cfg.padding_mode
        kw = cfg.kw
        norm = cfg.norm

        # Other settings
        self.num_class = num_class
        self.activation = nn.LeakyReLU()

        # If use normal GAN loss, last layer if FC and use sigmoid
        self.use_sigmoid = cfg.use_sigmoid if hasattr(cfg, 'use_sigmoid') else False

        # Layer structure
        self.c0_0 = conv_layer(2, top, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=norm, padding_mode=padding_mode)
        self.c1_0 = conv_layer(top, top*2, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=norm, padding_mode=padding_mode)
        self.c2_0 = conv_layer(top*2, top*4, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=norm, padding_mode=padding_mode)
        self.c3_0 = conv_layer(top*4, top*8, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=norm, padding_mode=padding_mode)
        if self.use_sigmoid:
            self.l_last = nn.Linear(top*frame_nums//2, 1) 
        else:
            if norm is not None and norm.startswith('spectral'):
                self.c_last = conv_layer(top*8, 1, ksize=(1,1), stride=(1,1), pad=(0,0), normalize='spectral', padding_mode=padding_mode)
            else:
                self.c_last = conv_layer(top*8, 1, ksize=(1,1), stride=(1,1), pad=(0,0), normalize=None, padding_mode=padding_mode)

        self.l_cls = nn.Linear(top*frame_nums//2, num_class)
        self.softmax = nn.Softmax(dim=1)

        # Spectral normalization
        if norm is not None and norm.startswith('spectral'):
            if num_class > 0:
                self.l_cls = spectral_norm(self.l_cls)
            if self.use_sigmoid:
                self.l_last = spectral_norm(self.l_last)

        # Initialize linear layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.02)
       

    def forward(self, x, remove_softmax=False):
        bs = x.shape[0]
 
        h = x
        h = self.activation(self.c0_0(h))
        h = self.activation(self.c1_0(h))
        h = self.activation(self.c2_0(h))
        h = self.activation(self.c3_0(h))

        if self.use_sigmoid:
            out = torch.sigmoid(self.l_last(h.view(bs, -1)))
        else:
            out = self.c_last(h)

        if remove_softmax:
            cls = self.l_cls(h.view(bs, -1))
        else:
            cls = self.softmax(self.l_cls(h.view(bs, -1)))

        return out, cls


    def inference(self, x):
        h = x
        h = self.activation(self.c0_0(h))
        h = self.activation(self.c1_0(h))
        return h




class LatentTransformation(nn.Module):
    def __init__(self, cfg, num_class):
        super().__init__()
    
        self.z_dim = cfg.z_dim
        self.w_dim = cfg.w_dim
        self.normalize_z = PixelNormalizationLayer() if cfg.normalize_z else None

        activation = nn.LeakyReLU()

        self.latent_transform = nn.Sequential(
                conv_layer(self.z_dim*2, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation,
                conv_layer(self.w_dim, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation,
                conv_layer(self.w_dim, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation,
                conv_layer(self.w_dim, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation,
                conv_layer(self.w_dim, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation
        )

        self.label_embed = nn.Embedding(num_class, self.w_dim)


    def forward(self, z, labels):
        labels_embed = self.label_embed(labels).view([-1, self.z_dim, 1, 1])
        z = torch.cat([z.view([-1,self.z_dim,1,1]), labels_embed], dim=1)

        if self.normalize_z is not None:
            z = self.normalize_z(z)
     
        w = self.latent_transform(z)
        return w 
