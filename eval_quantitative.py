#!/usr/bin/env python
import os
import sys
import glob
import re
import argparse
import time

import math
import pickle
import numpy as np
import pandas as pd
from scipy import interpolate
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from core import models
from core.datasets.dataset import BVHDataset
from core.utils.config import Config
from core.utils.motion_utils import reconstruct_v_trajectory, sampling, calcurate_footskate, calcurate_trajectory_error
from core.utils.bvh_to_joint import get_standard_format, cut_zero_length_bone


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Argument Parser 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def parse_args():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('config',   help='Config file path.')
    parser.add_argument('--weight', default=None,
                                    help='Path to generator weight. If not specified, use latest one')
    parser.add_argument('--gpu', default=0, type=int,
                                    help='GPU ID (negative value indicates CPU)')
    
    args = parser.parse_args()
    return args


#%---------------------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = Config.from_file(args.config)


    ## Set ?PU device
    cuda = torch.cuda.is_available()
    if cuda and args.gpu > -1:
        print('\033[1m\033[91m' + '# cuda available!' + '\033[0m')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'

    # Set up generator network
    num_class = len(cfg.train.dataset.class_list)
    gen = getattr(models, cfg.models.generator.model)(cfg.models.generator, num_class).to(device)

    # Load weight
    if args.weight is None:
        checkpoint_path = os.path.join(cfg.test.out, 'gen.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = sorted(glob.glob(os.path.join(cfg.test.out, 'checkpoint', 'iter_*.pth.tar')))[-1]
    else:
        checkpoint_path = args.weight

    if not os.path.exists(checkpoint_path):
        print('Generator weight not found!')
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


    ## Create name and embed w pair
    inputs = [] 
    z = gen.make_hidden(1,1).to(device) if cfg.models.generator.use_z else None
    for i, style in enumerate(cfg.train.dataset.class_list):
        label = torch.Tensor([i]).type(torch.LongTensor).to(device)
        w = gen.latent_transform(z, label) 
        inputs.append((style, w))

    # Each label corresponds to rows
    rows = ['Average'] + [s[0] for s in inputs]


    #======================================================================
    #
    ### Prepare target data
    # 
    #======================================================================
   
    print(f'Inference on  \033[1m{cfg.test.dataset.data_root}\033[0m')
    targets = []
    # Set up dataset
    test_dataset = BVHDataset(cfg.test.dataset, mode='test')
    # Prepare target data  
    for k in range(len(test_dataset)):
        x_data, control_data, _  = test_dataset[k]

        x_data = torch.from_numpy(x_data)
        control_data = torch.from_numpy(control_data) 

        # Convert tensor to batch
        x_data = x_data.unsqueeze(0).unsqueeze(1).type(torch.FloatTensor)
        control_data = control_data.unsqueeze(0).unsqueeze(1).type(torch.FloatTensor)
        original_curve = control_data[0,0,:,:]
        # Generate input velocity spline
        v_control = control_data[:,:,1:,] - control_data[:,:,:-1,:]
        v_control = F.pad(v_control, (0,0,1,0), mode='reflect')
        v_control = v_control.to(device)

        targets.append({'name':f'{k:03d}', 'x_data':x_data, 'v_control':v_control, 'original_curve':original_curve})


    # Each target data corresponds to columns
    columns = ['Average'] + [data['name'] for data in targets]




    #======================================================================
    #
    ###   Test Start
    #
    #======================================================================

    ## Define output directory
    test_dataset_name = os.path.split(cfg.test.dataset.data_root)[1]
    result_dir = f'{cfg.test.out}/eval/iter_{iteration}/{test_dataset_name}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    
    ## Testing option
    standard_bvh = cfg.test.dataset.standard_bvh if hasattr(cfg.test.dataset, 'standard_bvh') else 'core/datasets/CMU_standard.bvh'
    skelton, _, joints_to_index, _ = get_standard_format(standard_bvh)
    _, non_zero_joint_to_index = cut_zero_length_bone(skelton, joints_to_index)
    id_rightleg = non_zero_joint_to_index['RightToeBase'] 
    id_leftleg = non_zero_joint_to_index['LeftToeBase']     

    

    ## Evaluate each sample
    trajectory_error_data = np.zeros((len(rows), len(columns)))
    footskate_data = np.zeros((len(rows), len(columns)))
    for i, test_data in enumerate(targets):
        v_control = test_data['v_control']
        original_curve = test_data['original_curve']
        original_curve = original_curve - original_curve[0,:]

        result_dic = {}
        for j, (name, w) in enumerate(inputs): 
            start_time = time.time()
       
            #----------------------------------------
            #   Inference with model
            #----------------------------------------
            fake_v_trajectory, x_fake = gen(v_control, w=w)


            # Velocity to positon
            fake_trajectory = reconstruct_v_trajectory(fake_v_trajectory.data.cpu(), torch.zeros(1,1,1,3))
            x_fake = torch.cat((fake_trajectory, x_fake.cpu()), dim=3)

            # Denormalize
            x_fake *= cfg.test.dataset.scale

            #---------------------------------------------------
            #   Calcurate on metrics
            #---------------------------------------------------
            frame_length = x_fake.shape[2]
            # Calcurlate foot skating distance
            footskate_dist = calcurate_footskate(x_fake[0,0,:,:3], x_fake[0,0,:,id_rightleg*3:id_rightleg*3+3], x_fake[0,0,:,id_leftleg*3:id_leftleg*3+3]) / frame_length
            footskate_data[j+1,i+1] = round(footskate_dist, 6)
            # Calcurlate trajectory error
            error_dist = calcurate_trajectory_error(x_fake[0,0,:,:3], original_curve, 8, 32) 
            trajectory_error_data[j+1,i+1] = round(error_dist, 6)


    #---------------------------------------------------
    #   Merge all results
    #---------------------------------------------------
    for name, data in [('trajectory_erorr_dist', trajectory_error_data), ('footskate_dist', footskate_data)]:
        # Get average 
        data[:,0] = np.sum(data[:,1:], axis=1) / (len(columns)-1)
        data[0,:] = np.sum(data[1:,:], axis=0) / (len(rows)-1)
        data = data.tolist()
      
        # Save as csv  
        df = pd.DataFrame(data, index=rows, columns=columns)
        print(name, '\n', df)
        df.to_csv(os.path.join(result_dir, f'{name}.csv'))



if __name__ == '__main__':
    main()
