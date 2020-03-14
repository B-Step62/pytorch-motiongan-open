import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F



class SPADE(nn.Module):
    def __init__(self, in_ch, num_class, cfg):
        super().__init__()
        
        if cfg.param_free_norm == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(in_ch, affine=False)
        elif cfg.param_free_norm ==  'batch':
            self.param_free_norm = nn.BatchNorm2d(in_ch, affine=False)
        else:
            raise ValueError(f'{cfg.param_free_norm} is not a valid parm-free norm type in SPADE.')
      
        nhidden = cfg.nhidden

        kw = cfg.kw if hasattr(cfg, 'kw') else 5
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(num_class, nhidden,  kernel_size=(kw,1), padding=(kw//2,0)),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, in_ch, kernel_size=(kw,1), padding=(kw//2,0))
        self.mlp_beta = nn.Conv2d(nhidden, in_ch, kernel_size=(kw,1), padding=(kw//2,0))

    def forward(self, x, y):
        assert x.shape[2]==y.shape[2]
        # if y is 1D sequential input
        if y.shape[3] == 1:
            y = y.repeat(1,1,1,x.shape[3])
        # Generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
 
        # Produce scaling and bias conditioned on trajectory
        actv = self.mlp_shared(y)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized*(1+gamma) + beta
        return out
     

class AdaIN(nn.Module):
    def __init__(self, dim, w_dim):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-8
        self.scale_transform = nn.Conv2d(w_dim, dim, 1, 1, 0)
        self.bias_transform = nn.Conv2d(w_dim, dim, 1, 1, 0)

        nn.init.normal_(self.bias_transform.weight, mean=0., std=0.02)
        nn.init.zeros_(self.bias_transform.bias)
 
    def forward(self, x, w, w1=None, ratio=[0,1], interp_mode='inter'):
        x = F.instance_norm(x, eps=self.epsilon)
    
        scale = self.scale_transform(w)
        bias = self.bias_transform(w)

        # Style mixing
        if w1 is not None:
            scale1 = self.scale_transform(w1)
            bias1 = self.bias_transform(w1)
            if interp_mode == 'inter':
                scale = (ratio[0] * scale + ratio[1] * scale1) / (ratio[0]+ratio[1])
                bias = (ratio[0] * bias + ratio[1] * bias1) / (ratio[0]+ratio[1])
            elif interp_mode == 'extra':
                scale = (ratio[0] * scale - ratio[1] * scale1) / (ratio[0]-ratio[1])
                bias = (ratio[0] * bias - ratio[1] * bias1) / (ratio[0]-ratio[1])
            else:
                raise ValueError(f'Invalid interpolation mode {interp_mode}.')
    
        return scale * x + bias

    def eval_params(self, w):
        scale = self.scale_transform(w)
        bias = self.bias_transform(w)
        return {'scale':scale, 'bias':bias}

        

class MaskAdaIN(nn.Module):
    def __init__(self, dim, w_dim):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-8
        self.scale_transform = nn.Conv2d(w_dim, dim, 1, 1, 0)
        self.bias_transform = nn.Conv2d(w_dim, dim, 1, 1, 0)

        nn.init.normal_(self.bias_transform.weight, mean=0., std=0.02)
        nn.init.zeros_(self.bias_transform.bias)

    def mean2d(self, x):
        assert len(x.shape) == 4
        return torch.mean(torch.mean(x, dim=3, keepdim=True), dim=2, keepdim=True)
 
    def forward(self, x, mask, w, w1=None, ratio=[0,1], interp_mode='inter'):
        # Compute mean and variance except masked_region
        assert x.shape[0] == mask.shape[0]
        mean = self.mean2d(x * mask) / self.mean2d(mask)
        var = self.mean2d(((x - mean) * mask) **2) / self.mean2d(mask)

        # Normalize x
        x = (x - mean) / (self.epsilon + var).sqrt()

        scale = self.scale_transform(w)
        bias = self.bias_transform(w)

        # Style mixing
        if w1 is not None:
            scale1 = self.scale_transform(w1)
            bias1 = self.bias_transform(w1)
            if interp_mode == 'inter':
                scale = (ratio[0] * scale + ratio[1] * scale1) / (ratio[0]+ratio[1])
                bias = (ratio[0] * bias + ratio[1] * bias1) / (ratio[0]+ratio[1])
            elif interp_mode == 'extra':
                scale = (ratio[0] * scale - ratio[1] * scale1) / (ratio[0]-ratio[1])
                bias = (ratio[0] * bias - ratio[1] * bias1) / (ratio[0]-ratio[1])
            else:
                raise ValueError(f'Invalid interpolation mode {interp_mode}.')
    
        return (scale * x + bias) * mask

    def eval_params(self, w):
        scale = self.scale_transform(w)
        bias = self.bias_transform(w)
        return {'scale':scale, 'bias':bias}


class AdaIN_1D(nn.Module):
    def __init__(self, dim, w_dim):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-8
        self.scale_transform = nn.Conv1d(w_dim, dim, 1, 1, 0)
        self.bias_transform = nn.Conv1d(w_dim, dim, 1, 1, 0)

        nn.init.normal_(self.scale_transform.weight, mean=0., std=0.02)
        nn.init.zeros_(self.scale_transform.bias)
 
        nn.init.normal_(self.bias_transform.weight, mean=0., std=0.02)
        nn.init.zeros_(self.bias_transform.bias)
 
    def forward(self, x, w, debug=False):
        x = F.instance_norm(x, eps=self.epsilon)
    
        scale = self.scale_transform(w)
        bias = self.bias_transform(w)
    
        return scale * x + bias



class PixelNormalizationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
    
    def forward(self, x):
        x2 = x**2
        length_inv = torch.rsqrt(x2.mean(1, keepdim=True) + self.epsilon)
        return x * length_inv


class ConditionalInstanceNorm(nn.Module):
    pass

## used only for define CategoricalConditionalBatchNorm
class BatchNorm2d(nn.BatchNorm2d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

class CategoricalConditionalBatchNorm(nn.Module):
    # as in the chainer SN-GAN implementation, we keep per-cat weight and bias
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            weight = self.weight.index_select(0, cats).view(shape)
            bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


