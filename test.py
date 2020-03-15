import argparse
import os
import sys
import random
import time

import math
import pickle
import numpy as np
from scipy import interpolate
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from core import models
from core.datasets.dataset import BVHDataset
from core.utils.config import Config
from core.visualize.save_video import save_video 
from core.utils.motion_utils import reconstruct_v_trajectory



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Argument Parser 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_args():
    parser = argparse.ArgumentParser(description='EqualledCycleGAN')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--num_samples', type=int, default=2)
    args = parser.parse_args()
    return args


#%---------------------------------------------------------------------------------------
def test():
    global args, cfg, device

    args = parse_args()
    cfg = Config.from_file(args.config)


    # Set ?PU device
    cuda = torch.cuda.is_available()
    if cuda:
        print('\033[1m\033[91m' + '# cuda available!' + '\033[0m')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'


    #####################################################    
    ## Prepare for test 
    #####################################################  

    # Set up generator network
    num_class = len(cfg.train.dataset.class_list)
    gen = getattr(models, cfg.models.generator.model)(cfg.models.generator, num_class).to(device)

    total_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f'Total parameter amount : \033[1m{total_params}\033[0m')


    # Load weight
    if args.weight is not None:
        checkpoint_path = args.weight
    else:
        checkpoint_path = os.path.join(cfg.test.out, 'gen.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = sorted(glob.glob(os.path.join(cfg.test.out, 'checkpoint', 'iter_*.pth.tar')))[-1]

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


    # Set up dataset
    test_dataset = BVHDataset(cfg.test.dataset, mode='test')
    test_dataset_name = os.path.split(cfg.test.dataset.data_root.replace('*', ''))[1]


    # Set standard bvh
    standard_bvh = cfg.test.dataset.standard_bvh if hasattr(cfg.test.dataset, 'standard_bvh') else 'core/datasets/CMU_standard.bvh'


    # Create output directory
    result_dir = f'{cfg.test.out}/test/iter_{iteration}/{test_dataset_name}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    #####################################################    
    ## Test start
    #####################################################  
    for i in range(len(test_dataset)):
        x_data, control_data, label  = test_dataset[i]

        if x_data.shape[0] < cfg.train.dataset.frame_nums // cfg.train.dataset.frame_step:
            continue

        # Motion and control signal data
        x_data = torch.from_numpy(x_data).unsqueeze(0).unsqueeze(1).type(torch.FloatTensor)
        x_real = Variable(x_data).to(device)

        control_data = torch.from_numpy(control_data).unsqueeze(0).unsqueeze(1).type(torch.FloatTensor) 
        control = control_data.to(device)

        # Convert root trajectory to verocity
        gt_trajectory = x_data[:,:,:,0:3]
        gt_v_trajectory = gt_trajectory[:,:,1:,:] - gt_trajectory[:,:,:-1,:]
        gt_v_trajectory = F.pad(gt_v_trajectory, (0,0,1,0), mode='reflect')
        gt_v_trajectory = Variable(gt_v_trajectory).to(device)


        # Convert control curve to velocity 
        v_control = control[:,:,1:,] - control[:,:,:-1,:]
        v_control = F.pad(v_control, (0,0,1,0), mode='reflect')
        v_control = Variable(v_control).to(device)


        results_list = []
        start_time = time.time()

        # Generate fake sample
        for k in range(args.num_samples):
            # Generate noize z
            z = gen.make_hidden(1, x_data.shape[2]).to(device) if cfg.models.generator.use_z else None
            fake_label = torch.randint(0, len(cfg.train.dataset.class_list), size=(1,)).type(torch.LongTensor).to(device)

            fake_v_trajectory, x_fake = gen(v_control, z, fake_label)
            fake_trajectory = reconstruct_v_trajectory(fake_v_trajectory.data.cpu(), torch.zeros(1,1,1,3))
 
            caption = f'{cfg.train.dataset.class_list[fake_label]}_{k}'
            results_list.append({'caption': caption, 'motion': torch.cat((fake_trajectory, x_fake.data.cpu()), dim=3), 'control': control.data.cpu()})

        avg_time = (time.time() - start_time) / args.num_samples

        # Save results
        result_path = result_dir + f'/{i:03d}.avi'
        print(f'\nInference : {result_path} ({v_control.shape[2]} frames) Time: {avg_time:.05f}') 
        save_video(result_path, results_list, cfg.test)



if __name__ == '__main__':
    test()
