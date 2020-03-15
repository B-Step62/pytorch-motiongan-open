import argparse
import os
import sys
import glob
import re

import numpy as np
import torch

from core import models
from core.utils.config import Config
from core.statistics.tSNE import apply_tSNE
from core.statistics.pca import apply_pca
from core.statistics.distance_heatmap import draw_heatmap


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Argument Parser 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_args():
    parser = argparse.ArgumentParser(description='EqualledCycleGAN')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--mode', choices=['pca', 'tSNE', 'heatmap'], required=True)
    parser.add_argument('--target', default='w',
                        help='Subject to analize. It can be chosen from "w", "adain_???_".')
    parser.add_argument('--weight', type=str)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--components', '-c', type=str, default='1-2',
                        help='Specify component to show on 2d-axis in form of "x1-y1,x2-y2,x3-y3,...". Each pair of components are shown in separate graph in one file.')
    args = parser.parse_args()
    return args


#%---------------------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = Config.from_file(args.config)

    # Set ?PU device
    cuda = torch.cuda.is_available()
    if cuda:
        print('\033[1m\033[91m' + '# cuda available!' + '\033[0m')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'


    ## Set up generator network
    num_class = len(cfg.train.dataset.class_list)
    gen = getattr(models, cfg.models.generator.model)(cfg.models.generator, num_class).to(device)



    ## Load weight
    if args.weight is not None:
        checkpoint_path = args.weight
    else:
        checkpoint_path = os.path.join(cfg.test.out, 'gen.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = sorted(glob.glob(os.path.join(cfg.test.out, 'checkpoint', 'iter_*.pth.tar')))[-1]

    if not os.path.exists(checkpoint_path):
        print('Generator weight not found!')
        sys.exit()
    else:
        print(f'Loading generator model from \033[1m{checkpoint_path}\033[0m')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'gen_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['gen_state_dict'])
            iteration = checkpoint['iteration']
        else:
            gen.load_state_dict(checkpoint)
            iteration = cfg.train.total_iterations 
    gen.eval()


    ## Create output directory
    result_dir = f'{cfg.test.out}/statistics/iter_{iteration}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    ## Parse component list (for only PCA and tSNE)
    components = []
    for pattern in re.findall(r'(\d+-\d+)', args.components):
        pair = pattern.split('-')
        components.append([int(pair[0]), int(pair[1])])
    


    ## Conduct analisis

    # Apply PCA
    if args.mode == 'pca':
        result_path = os.path.join(result_dir, f'PCA_{args.target}.pdf')
        apply_pca(cfg, gen, result_path, args.target, components, device)

    # Apply tSNE
    elif args.mode == 'tSNE':
        result_path = os.path.join(result_dir, f'tSNE_{args.target}.pdf')
        apply_tSNE(cfg, gen, result_path, args.target, components, device)

    # Calcurate distance between centroid of each cluster and visualize as heatmap on matrix
    elif args.mode=='heatmap':    
        assert args.target == 'w'
        draw_heatmap(cfg, gen, result_dir, device) 

    else:
        raise ValueError('Invalid mode!')



if __name__ == '__main__':
    main()
