#!/usr/bin/env python
import os
import sys
import glob
import re
import argparse
import time
import random
import math
import pickle

import numpy as np
from scipy import interpolate
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from core import models
from core.datasets.dataset import BVHDataset
from core.utils.config import Config
from core.visualize.save_video import save_video
from core.visualize.save_timelapse import save_timelapse
from core.utils.motion_utils import reconstruct_v_trajectory, sampling, convert_event_to_spline


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Argument Parser 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def parse_args():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('--out', default=None,
                                    help='Output directory')
    parser.add_argument('--weight', default=None,
                                    help='Path to generator weight. If not specified, use latest one')
    parser.add_argument('--style_mode', choices=['single','mix','operation','random'], default='random', 
                                    help='Type of inference.')
    parser.add_argument('--style', default='', 
                                    help='Specify inference style. If blank, all style inferenced.')
    parser.add_argument('--num_samples', default=4, type=int, 
                                    help='Number of inference style for style-mode : random.')
    parser.add_argument('--model', choices=['MotionGAN','TCN','LSTM','ACGAN'], default='MotionGAN', 
                                    help='Model architecture')

    parser.add_argument('--target', choices=['dataset','draw'], default='dataset',
                                    help='Type of target data')
    parser.add_argument('--target_file', default='', 
                                    help='specify target .pkl or .bvh file')
    parser.add_argument('--control_point', '-cp', default=None, type=int,
                                    help='number of control pointfor spline interpolation')
    parser.add_argument('--splinex', '-x', default=1, type=float)
    parser.add_argument('--speed', '-sp', default=8, type=int)
    parser.add_argument('--fps', default=1.0, type=float)
    parser.add_argument('--start_frame', '-st', default=0, type=int, 
                                    help='Start frame index of video')

    parser.add_argument('--gpu', default=0, type=int,
                                    help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--save_pics', action='store_true',
                                    help='specify if you want to save time-lapse images')
    parser.add_argument('--hide_curve', action='store_true',
                                    help='specify if you dont want to show original input curve in video')
    parser.add_argument('--save_separate', action='store_true',
                                    help='specify if you dont want to save all sample within one video')
    parser.add_argument('--save_format', choices=['avi', 'gif'], default='avi',
                                    help='video format')
    parser.add_argument('--azim', default=45, type=int,
                                    help='Horizontal angle of video')
    parser.add_argument('--elev', default=10, type=int,
                                    help='Vertical angle of video')
    
    args = parser.parse_args()
    return args


#%---------------------------------------------------------------------------------------

def make_video():
    args = parse_args()
    cfg = Config.from_file(args.config)


    #======================================================================
    #
    ### Set up model
    # 
    #======================================================================

    ## Set ?PU device
    cuda = torch.cuda.is_available()
    if cuda and args.gpu > -1:
        print('\033[1m' + '# cuda available!' + '\033[0m')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'

    ## Define Generator
    gen = getattr(models, cfg.models.generator.model)(cfg.models.generator, len(cfg.train.dataset.class_list)).to(device)

    # Load weight
    if args.weight is None:
        checkpoint_path = os.path.join(cfg.test.out, 'gen.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = sorted(glob.glob(os.path.join(cfg.test.out, 'checkpoint', 'iter_*.pth.tar')))[-1]
    else:
        checkpoint_path = args.weight

    if not os.path.exists(checkpoint_path):
        print('\033[31m' + 'generator weight not found!' + '\033[0m')
    else:
        print('\033[33m' + 'loading generator model from ' + '\033[1m' + checkpoint_path + '\033[0m')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'gen_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['gen_state_dict'])
            iteration = checkpoint['iteration']
        else:
            gen.load_state_dict(checkpoint)
            iteration = cfg.train.total_iterations
    gen.eval()



    #======================================================================
    #
    ### Label Embedding 
    # 
    #======================================================================
   
    '''
    * Style Mode
       - single : Inference with each style
       - random : Randomly chosen 4 style
       - mix : Interpolation between two style
       - operation : Addition and subtraction with multiple style
    '''

    ## Define label embed function
    if args.model in ['MotionGAN', 'TCN']:
        z = 0 * gen.make_hidden(1, 1).to(device) if cfg.models.generator.use_z else None
        LabelEmbedder = lambda style: gen.latent_transform(z, style_to_label(style))
    elif args.model == 'ACGAN':
        z = 0 * gen.make_hidden(1, 1).to(device) if cfg.train.generator.use_z else None
        LabelEmbedder = lambda style: gen.label_emb(style_to_label(style))
    elif args.model == 'LSTM':
        LabelEmbedder = lambda style: gen.label_emb(style_to_label(style))
    else:
        pass

    # Define style name convert function
    def style_to_label(style):
        return torch.Tensor([cfg.train.dataset.class_list.index(style)]).type(torch.LongTensor).to(device) 



    ## Create name and embed w pair
    inputs = [] 

    #  Mode :  Single style inference
    if args.style_mode == 'single':
        style_list = args.style.split(',') if args.style else cfg.train.dataset.class_list 
        for style in style_list:
            w = LabelEmbedder(style)
            inputs.append((style, w))

    #  Mode :  Random style inference
    if args.style_mode == 'random':
        style_list = cfg.train.dataset.class_list 
        for style in random.sample(style_list, args.num_samples):
            w = LabelEmbedder(style)
            inputs.append((style, w))

    #  Mode :  Style Mix
    elif args.style_mode == 'mix':
        style1, style2 = args.style.split('-')
        w1 = LabelEmbedder(style1)
        w2 = LabelEmbedder(style2)
        inputs.append((style1, w1))
        for ratio in [0.5]:
            w = ratio * w1 + (1.0 - ratio) * w2
            inputs.append(('Mix'+str(ratio), w))
        inputs.append((style2, w2))

    #  Mode :  Style Operation
    elif args.style_mode == 'operation':
        def operation_parser(op_str):
            operands, operators = [], ['+']
            tail = 0 
            for i in range(len(op_str)):
                if op_str[i] in ['+', '-']:
                    operands.append(op_str[tail:i])
                    operators.append(op_str[i])
                    tail = i+1 
            operands.append(op_str[tail:])
            assert len(operands) == len(operators)
            return operands, operators
        operands, operators = operation_parser(args.style)
        # Embed first operand
        w_result = 0
        # Embed rest operands and calcurate operation 
        for operand, operator in zip(operands, operators):
            w = LabelEmbedder(operand)
            inputs.append((operand,w))
            if operator == '+':
                w_result += w
            elif operator == '-':
                w_result -= w
            else:
                raise ValueError('Invalid operator {operator}')
        inputs.append((args.style, w_result))
        


    #======================================================================
    #
    ### Define target data
    # 
    #======================================================================
   
    '''
       - dataset : Inference on test dataset specified in config
       - draw : Inference on draw curve (.pkl file)
    '''
    targets = []

    ### Mode : Inference on test dataset
    if args.target == 'dataset':
        print(f'\033[92mInference on  \033[1m{cfg.test.dataset.data_root}\033[0m')

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
            original_curve = control_data[0,0,:,:].data.numpy() if not args.hide_curve else None
            # Generate input velocity spline
            v_control = control_data[:,:,1:,] - control_data[:,:,:-1,:]
            v_control = F.pad(v_control, (0,0,1,0), mode='reflect')
            v_control = v_control.to(device)
 
            targets.append({'name':f'{k:03d}', 'x_data':x_data, 'v_control':v_control, 'original_curve':original_curve})

    ### Mode : Inference on draw curve
    elif args.target == 'draw':
        print(f'\033[92mInference on  \033[1m{args.target_file}\33[0m')

        # Open .pkl file
        assert (args.target_file).endswith('.pkl')
        with open(args.target_file, mode='rb') as f:
            data = pickle.load(f)
            if 'trajectory' in data:
                trajectory_raw = data['trajectory']
                control_point_interval = len(trajectory_raw) // args.control_point if args.control_point is not None else 120
                control_data = convert_event_to_spline(trajectory_raw, control_point_interval=control_point_interval)
                data['control'] = control_data
            else:
                control_data = data['spline']
                trajectory_raw = []
            data_name = os.path.splitext(os.path.split(args.target_file)[1])[0]
            speed_modify = args.speed

        frame_step = int(cfg.train.frame_step // args.splinex * args.fps)
        control_data = torch.from_numpy(control_data).unsqueeze(0).unsqueeze(1).type(torch.FloatTensor)
        speed_modify /= scale

        batch_length = int(control_data.shape[2]/(frame_step*args.fps)) // 64 * 64
        control_data = F.interpolate(control_data, size=(batch_length, spline_data.shape[3]), mode='bilinear', align_corners=True)
        control_data *= args.splinex / speed_modify
        if args.hide_curve:
            original_curve = None
        else:
            original_curve = (control_data[0,0,:,:] - control_data[0,0,0,:]).data.numpy()
            original_curve[:,1] = original_curve[:,1] - original_curve[:,1]

        # Convert position to velocity
        v_control = control_data[:,:,1:,] - control_data[:,:,:-1,:]
        v_control = F.pad(v_control, (0,0,1,0), mode='reflect')
        v_control = Variable(v_control).to(device)

        targets.append({'name': data_name, 'v_control':v_control, 'original_curve':original_curve})





    #======================================================================
    #
    ###   Test Start
    #
    #======================================================================

    ## Define output directory
    if args.target == 'dataset':
        test_dataset_name = os.path.split(cfg.test.dataset.data_root)[1]
    elif args.target == 'draw':
        test_dataset_name = 'Draw' 
    result_dir_top = f'{cfg.test.out}/test/iter_{iteration}/{test_dataset_name}' if args.out is None else os.path.join(args.out, test_dataset_name)
    
    if not os.path.exists(result_dir_top):
        os.makedirs(result_dir_top)

    
    ## Testing option
    standard_bvh = cfg.test.dataset.standard_bvh if hasattr(cfg.test.dataset, 'standard_bvh') else 'core/datasets/CMU_standard.bvh'
    # Prepare initial pose for lstm
    if args.model == 'LSTM':
        if 'initial_pose' in checkpoint:
            initial_pose = checkpoint['initial_pose']
        else:
            initial_pose = np.load('core/utils/initial_post_lstm.npy')
            initial_pose = torch.from_numpy(initial_pose).view(1,1,1,-1).type(torch.FloatTensor)
    
    ## Generate each sample
    for test_data in targets:
        v_control = test_data['v_control']
        original_curve = test_data['original_curve']

        result_list = []
        for name, w in inputs: 
            start_time = time.time()
            original_curve_j = original_curve.copy()
       
            #----------------------------------------
            #   Inference with model
            #----------------------------------------
            ## MotionGAN
            if args.model == 'MotionGAN':
                fake_v_trajectory, x_fake_motion = gen(v_control, w=w)
            ## ACGAN
            elif args.model == 'ACGAN':
                fake_v_trajectory, x_fake_motion = gen(v_control, z.repeat(1,1,v_spline.shape[2],1), label_embed=w)
            ## TCN
            elif args.model == 'TCN':
                # Inference each 128 frames
                for t in range(0, v_spline.shape[2], 128):
                    fake_v_trajectory_t, x_fake_motion_t = gen(v_control[:,:,t:t+128,:], w=w)
                    fake_v_trajectory = fake_v_trajectory_t if t==0 else torch.cat((fake_v_trajectory, fake_v_trajectory_t), dim=2)
                    x_fake_motion = x_fake_motion_t if t==0 else torch.cat((x_fake_motion, x_fake_motion_t), dim=2)
            ## LSTM
            elif args.model == 'LSTM':
                traj_t, pose_t = v_control[:,:,:1,:], initial_pose.to(device)
                for t in range(v_control.shape[2]):
                    traj_t, pose_t = gen(v_control[:,:,t,:], pose_t, traj_t, label_embed=w)
                    fake_v_trajectory = traj_t if t==0 else torch.cat((fake_v_trajectory, traj_t), dim=2)
                    x_fake_motion = pose_t if t==0 else torch.cat((x_fake_motion, pose_t), dim=2)

            #---------------------------------------------------
            #   Convert model output to viewable joint position
            #---------------------------------------------------
            if x_fake_motion.shape[2] > args.start_frame:
                x_fake_motion = x_fake_motion[:,:,args.start_frame:,:]
                fake_v_trajectory = fake_v_trajectory[:,:,args.start_frame:,:]
                if original_curve_j is not None:
                    original_curve_j = original_curve_j[args.start_frame:,:] - original_curve_j[args.start_frame,:]
            else:
                if original_curve_j is not None:
                    original_curve_j = original_curve_j - original_curve_j[0,:]



            # Root position at start frame
            start_position = torch.zeros(1,1,1,3)
            if re.search(r"OldMan|Sneaky|Scared|Chicken|Dinosaur", name) is not None:
                start_position[0,0,0,1] = 15.0 / cfg.test.dataset.scale
            else:
                start_position[0,0,0,1] = 17.0 / cfg.test.dataset.scale

            # Velocity to positon
            fake_trajectory = reconstruct_v_trajectory(fake_v_trajectory.data.cpu(), start_position)
            x_fake = torch.cat((fake_trajectory, x_fake_motion.cpu()), dim=3)
            result_list.append({'caption': name, 'motion': x_fake.detach()[0,:,:,:], 'control': original_curve_j}) 

 
            # Measure time 
            avg_time = (time.time() - start_time)
    
            #------------------------------------------------
            #   Save each sample  
            #------------------------------------------------
            if args.save_separate:
                if args.style_mode == 'single':
                    result_dir = os.path.join(result_dir_top, name)
                else:
                    result_dir = os.path.join(result_dir_top, args.style)

                if not os.path.exists(result_dir):
                    os.mkdir(result_dir)
                result_path = os.path.join(result_dir, test_data['name']+'_'+name)
                print(f'\nInference : {result_path} {x_fake[0,0,:,:].shape}  Time: {avg_time:.05f}') 
                # Save result data
                with open(result_path+'.pkl', 'wb') as f:
                    pickle.dump(result_dic, f)
                # Save video
                save_video(result_path+'.'+args.save_format, result_list, cfg.test, camera_move='stand', elev=args.elev, azim=args.azim)
                if args.save_pics:
                    save_timelapse(result_path+'.png', result_list, cfg.test)
                result_list = []

        #------------------------------------------------
        #   Save all frame in one video
        #------------------------------------------------
        if not args.save_separate:
            result_dir = os.path.join(result_dir_top, args.style)
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            result_path = os.path.join(result_dir, test_data['name'])
            print(f'\nInference : {result_path} {x_fake[0,0,:,:].shape}  Time: {avg_time:.05f}') 
            # Save video
            save_video(result_path+'.'+args.save_format, result_list, cfg.test, camera_move='stand', elev=args.elev, azim=args.azim)
            if args.save_pics:
                save_timelapse(result_path+'.png', result_list, cfg.test)



if __name__ == '__main__':
    make_video()
