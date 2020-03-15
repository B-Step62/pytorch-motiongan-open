#!/usr/bin/env python
import argparse
import os
import sys
import glob
import shutil
import pickle

import numpy as np
from scipy import interpolate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn

from core import models
from core.datasets.dataset import create_data_from_npy
from core.utils.config import Config
from core.utils import motion_utils
from core.utils.bvh_to_joint import get_standard_format
from core.visualize.save_video import save_video



PRINT_INTERVAL = 100
PLOT_INTERVAL = 10
TOTAL_ITERATION = 2000
MAX_LR = 0.0005
LR_LINEAR_INCRESE_ITERATION = 50
LR_ANNEALING_START_ITERATION = 50

LAM_D_LOSS = 1.0
LAM_PRCP_LOSS = 0.005
LAM_REC_LOSS = 0.01


class w_Generator(nn.Module):
    def __init__(self, init_tensor, device):
        super(w_Generator, self).__init__()
        self.w = torch.nn.Parameter(init_tensor)

    def forward(self):
        return self.w

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Argument Parser 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path')
    parser.add_argument('--target', type=str, required=True) # To specify with wildcard '*', put '\' before asterisk to avoid error, like 'data/\*.pkl'
    parser.add_argument('--weight', type=str)

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--save_video', action='store_true')
    args = parser.parse_args()
    return args

def train():
    global args, cfg, device

    args = parse_args()
    cfg = Config.from_file(args.config)


    print('GPU: {}'.format(args.gpu))
    print('# iteration: {}'.format(TOTAL_ITERATION))
    print('')


    #===========================================================
    #
    ### Set up pre-trained TRUNet model
    # 
    #===========================================================

    # Set device
    cuda = torch.cuda.is_available()
    if cuda:
        print('# cuda available!')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'

    # set start iteration
    iteration = 0

    # Set up a neural network to train
    class_list = cfg.train.dataset.class_list
    num_class = len(class_list)
    gen = getattr(models, cfg.models.generator.model)(cfg.models.generator, num_class).to(device)
    dis = getattr(models, cfg.models.discriminator.model)(cfg.models.discriminator, cfg.train.dataset.frame_nums//cfg.train.dataset.frame_step, num_class).to(device)
    for param in gen.parameters():
        param.requires_grad = False
    for param in dis.parameters():
        param.requires_grad = False


    # Load weight
    if args.weight is not None:
        checkpoint_path = args.weight
    else:
        checkpoint_path = os.path.join(cfg.test.out, 'gen.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = sorted(glob.glob(os.path.join(cfg.test.out, 'checkpoint', 'iter_*.pth.tar')))[-1]

    if not os.path.exists(checkpoint_path):
        print('generator weight not found!')
    else:
        print('loading generator model from ' + checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        #for k, v in checkpoint['gen_state_dict'].items():
        #    print(k, type(v), v.device)
        if 'gen_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['gen_state_dict'])
            dis.load_state_dict(checkpoint['dis_state_dict'])
        else:
            gen.load_state_dict(checkpoint)
            dis.load_state_dict(torch.load(checkpoint_path.replace('gen.pth', 'dis.pth'), map_location=device))
    

    # Rearrange accorfing to fps
    frame_step = int(cfg.train.dataset.frame_step * args.fps)
    frame_nums = int(cfg.train.dataset.frame_nums * args.fps)



    ### Set train setting 
    # Standard skelton
    standard_bvh = cfg.train.dataset.standard_bvh if hasattr(cfg.train.dataset, 'standard_bvh') else 'core/datasets/CMU_standard.bvh'
    skeleton, non_end_bones, joints_to_index, _ = get_standard_format(standard_bvh)

    # Set Criterion
    criterion = torch.nn.MSELoss().to(device)
    


    #===========================================================
    #
    ###  Optimizing with each target data
    # 
    #===========================================================

    target_list = sorted(glob.glob(args.target))
    for target in target_list:


        #--------------------------------------------------
        #  Load target motion data    
        #--------------------------------------------------
        if target.endswith('.pkl'):
            with open(target, mode='rb') as f:
                data = pickle.load(f)

        elif target.endswith('npy'):
            control_point_interval = cfg.train.dataset.control_point_interval
            # Create path to save inclusive data (.pkl)
            top, name = os.path.split(target)
            name, ext = os.path.splitext(name)
            data_dir = os.path.join(top, f'processed_cp{control_point_interval}')
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            data_path = os.path.join(data_dir, name+'.pkl')
            data = create_data_from_npy(cfg.train.dataset, target, data_path, skeleton, joints_to_index)

        else:
            raise ValueError(f'Invalid file format. : {target}')
        

        motion_all = data['motion']
        motion_all = motion_all[20:,:] # Remove beggining frames
        motion_all = motion_all[:motion_all.shape[0]-motion_all.shape[0]%(frame_step*16):frame_step,:]
        motion_all /= cfg.train.dataset.scale

        control_all = motion_utils.sampling(data['trajectory'], data['spline_f'], data['spline_length_map'], 0.1, startT=0, endT=motion_all.shape[0]*frame_step, step=frame_step, with_noise=False)
        control_all[:,1] = np.zeros(control_all.shape[0])
        control_all /= cfg.train.dataset.scale
    

        #--------------------------------------------------
        #  Create output dir
        #--------------------------------------------------

        target_name = os.path.splitext(os.path.split(target)[1])[0]
        if args.fps != 1.0: target_name += f'_{args.fps}'
        if not os.path.exists(os.path.join(cfg.train.out, 'style_search', target_name)):
            os.makedirs(os.path.join(cfg.train.out, 'style_search', target_name))
        target_style = (os.path.splitext(os.path.split(target)[1])[0]).split('_')[0]
        target_style = target_style if target_style in class_list else 'unknown'


        #--------------------------------------------------
        #  Devide into fix length 
        #--------------------------------------------------
        w_gen_log_list = []
        for start_frame in range(0, motion_all.shape[0], frame_nums//frame_step):
            motion = motion_all[start_frame:start_frame+frame_nums//frame_step,:]
            if motion.shape[0] < frame_nums//frame_step: break
            control = control_all[start_frame:start_frame+frame_nums//frame_step,:]
  
        
            x_data = torch.from_numpy(motion).unsqueeze(0).unsqueeze(1).type(torch.FloatTensor)
            x_real = Variable(x_data).to(device)
    
            control = torch.from_numpy(control).unsqueeze(0).unsqueeze(1).type(torch.FloatTensor)
            control = control.to(device)
      
            #--------------------------------------------------
            #  Preprocessing motion data 
            #--------------------------------------------------

            # Convert trajectory to verocity
            # *_map is tiled tensor of verocity to inputting D (It must be same size as motion.)
            gt_trajectory = x_data[:,:,:,0:3]
            gt_v_trajectory = gt_trajectory[:,:,1:,:] - gt_trajectory[:,:,:-1,:]
            gt_v_trajectory = F.pad(gt_v_trajectory, (0,0,1,0), mode='reflect')
            gt_v_trajectory = Variable(gt_v_trajectory).to(device)
    
            # Generate input velocity from spline (input curve)
            v_control = control[:,:,1:,] - control[:,:,:-1,:]
            v_control = F.pad(v_control, (0,0,1,0), mode='reflect')
            v_control = Variable(v_control).to(device) 
    


            #--------------------------------------------------
            # Initialize model
            #--------------------------------------------------

            ## Intitial Tensor W 
            for c in range(num_class):
                label_c = torch.from_numpy(np.array([[c]]).astype(np.int64)).to(device)
                z = gen.make_hidden(1, cfg.train.dataset.frame_nums//cfg.train.dataset.frame_step).to(device) if cfg.models.generator.use_z else None
                # Generate w and embed from label
                w_c = gen.latent_transform(z, label_c)
                init_w = w_c if c==0 else torch.cat((init_w, w_c), dim=0)
            init_w = torch.mean(init_w, dim=0, keepdim=True)

            ## Create w model
            w_gen = w_Generator(init_w[0,:,:,0], device).to(device)
            opt_w = torch.optim.Adam(w_gen.parameters(), lr=MAX_LR, betas=(0.5, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_w, TOTAL_ITERATION - LR_ANNEALING_START_ITERATION, eta_min=0., last_epoch=-1.)
            # Switch model mode to train
            w_gen.train()

            #--------------------------------------------------
            #  Start optimizing 
            #--------------------------------------------------
            print(f'## Start Training (target : {target_name}, frame : {start_frame}-{start_frame+frame_nums//frame_step}) ###')
        
            loss_collector = {}
            w_gen_log = [w_gen()[:,0].data.cpu().numpy()]
            ## Training iteration
            for i in range(TOTAL_ITERATION):
                w = w_gen()
                fake_v_trajectory, x_fake = gen(v_control, w=w)
        
        
                # D input
                n_joints = (x_real.shape[3]-3)//3
                input_d_fake = torch.cat((fake_v_trajectory.repeat(1,1,1,n_joints).detach(), x_fake.detach()), dim=1)
                input_d_real = torch.cat((gt_v_trajectory.repeat(1,1,1,n_joints).detach(), x_real[:,:,:,3:]), dim=1)
                d_fake_adv, d_fake_cls = dis(input_d_fake, remove_softmax=True)
                d_real_adv, d_real_cls = dis(input_d_real, remove_softmax=True)
        
                # D loss
                d_loss = LAM_D_LOSS * criterion(d_fake_cls, d_real_cls)
                loss_collector['d_loss'] = LAM_D_LOSS * d_loss.item()
        
                # Percep loss
                prcp_loss = LAM_PRCP_LOSS * criterion(dis.inference(input_d_fake), dis.inference(input_d_real))
                loss_collector['prcp_loss'] = LAM_PRCP_LOSS * prcp_loss.item()
        
                # reconstrunction loss
                rec_loss = LAM_REC_LOSS * criterion(fake_v_trajectory, gt_v_trajectory) + criterion(x_fake, x_real[:,:,:,3:])
                loss_collector['rec_loss'] = LAM_REC_LOSS * rec_loss.item()
                     
                loss = rec_loss + d_loss + prcp_loss
                     
                opt_w.zero_grad()
                loss.backward()
                opt_w.step()
        
        
                
                # adjust learning rate
                if i+1 <= LR_LINEAR_INCRESE_ITERATION:
                    w_lr = MAX_LR * (i+1) / LR_LINEAR_INCRESE_ITERATION
                    for param_groups in opt_w.param_groups:
                        param_groups['lr'] = w_lr
                #elif (i+1) % LR_DECAY_INTERVAL == 0:
                #    w_lr /= 2.0
                #    for param_groups in opt_w.param_groups:
                #        param_groups['lr'] = w_lr
                elif i+1 > LR_ANNEALING_START_ITERATION:
                    scheduler.step()
    
                
        
                # print Log
                if (i + 1) % PRINT_INTERVAL == 0:
                    loss_summary = ''.join([f'{name}:{val:.5f}  ' for name, val in loss_collector.items()])
                      
                    cur_lr = opt_w.param_groups[0]['lr']
                    print((f'Iteration:[{i}][{TOTAL_ITERATION}]\t'
                           f'Loss {loss_summary}\t LR {cur_lr:.06f}'
                           ))
        
                # Store data for pca plot
                if (i+1) % PLOT_INTERVAL == 0:
                    w_gen_log.append(w[:,0].data.cpu().numpy())
        
            w_gen_log_list.append(w_gen_log)
    
            #--------------------------------------------------
            #  Save single frame subset results
            #--------------------------------------------------
    
            # Plot pca
            plot_on_PCA([w_gen_log], gen, class_list, os.path.join(cfg.train.out, 'style_search', target_name, f'PCA_iter_{i+1}_{start_frame}-{start_frame+frame_nums//frame_step}.pdf'))
        
            
            # Save preview image
            if args.save_video:
                save_path = os.path.join(cfg.train.out, 'style_search', target_name, f'iter_{TOTAL_ITERATION}_{start_frame}-{start_frame+frame_nums//frame_step}.avi')
                result_list = []
                w_fake = torch.from_numpy(w_gen_log[-1]).to(device)
                fake_v_trajectory, x_fake = gen(v_control[:1,:,:,:], w=w_fake)
                fake_trajectory = motion_utils.reconstruct_v_trajectory(fake_v_trajectory.data.cpu()[:1,:,:,:], gt_trajectory[:1,:,:1,:])
                result_list.append({'caption': 'found_w', 'motion': torch.cat((fake_trajectory, x_fake.data.cpu()[:1,:,:,:]), dim=3), 'control': control.data.cpu()[:1,:,:,:]})
            
                if target_style != 'unknown':
                    label_target = torch.from_numpy(np.array([[class_list.index(target_style)]]).astype(np.int64)).to(device)
                    w_target = gen.latent_transform(z, label_target)
                    target_v_trajectory, x_target = gen(v_control[:1,:,:,:], w=w_target)
                    target_trajectory = motion_utils.reconstruct_v_trajectory(target_v_trajectory.data.cpu()[:1,:,:,:], gt_trajectory[:1,:,:1,:])
                    result_list.append({'caption': target_style, 'motion': torch.cat((target_trajectory, x_target.data.cpu()[:1,:,:,:]), dim=3), 'control': control.data.cpu()[:1,:,:,:]})
            
                result_list.append({'caption': 'target', 'motion': x_data[:1,:,:,:], 'control': control.data.cpu()[:1,:,:,:]})
            
                save_video(save_path, result_list, cfg.test)
        
            #--------------------------------------------------
            #  Save full sequense results
            #--------------------------------------------------
            # Plot pca
            plot_on_PCA(w_gen_log_list, gen, class_list, os.path.join(cfg.train.out, 'style_search', target_name, f'PCA_iter_{i+1}_all.pdf'))
        


def plot_on_PCA(w_gen_log_list, gen, class_list, result_path):
    gen.eval()
 
    #=================================================-
    ### Plot training class
    #=================================================-
    num_class = len(class_list)

    data, class_data = [], []
    for c in range(num_class):
        label_c = torch.from_numpy(np.array([[c]]).astype(np.int64)).to(device)
        z = gen.make_hidden(1, cfg.train.dataset.frame_nums//cfg.train.dataset.frame_step).to(device) if cfg.models.generator.use_z else None
        # Generate w and embed from label
        w_c = gen.latent_transform(z, label_c)
        data.append(w_c[0,:,0,0].data.cpu().numpy())
        class_data.append(c)


    # Fit PCA and map train class
    data = np.array(data)
    pca = PCA(n_components=2, random_state=8)
    pca.fit(data)
    data_reduced = pca.transform(data)


    # Scatter points
    fig = plt.figure(figsize=(10, 10), dpi=216)
    plt.scatter(data_reduced[:,0], data_reduced[:,1], s=30, c=[cm.hsv(cl/len(class_list)) for cl in class_data], alpha=0.7)
    for c in range(len(class_list)):
        center = data_reduced[c]
        plt.text(center[0], center[1], class_list[c], fontsize=8, alpha=0.7)
 
    #=================================================-
    ### Plot obtained w (optimization history)
    #=================================================-
    for t, w_gen_log in enumerate(w_gen_log_list):
        w_reduced = pca.transform(np.array(w_gen_log))

        plt.plot(w_reduced[0:1,0], w_reduced[0:1,1], ms=12, c='black', marker='$S$') #Last one
        plt.plot(w_reduced[:,0], w_reduced[:,1], ms=2, c=cm.hsv(t/len(w_gen_log_list)), marker='.', alpha=0.5)
        plt.plot(w_reduced[-2:-1,0], w_reduced[-2:-1,1], ms=12, c='black', marker='$G$') #Last one


    # Save
    plt.savefig(result_path)
    plt.close()


if __name__ == '__main__':
    train()
