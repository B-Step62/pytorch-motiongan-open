#!/usr/bin/env python
import os
import sys
import math
import time
import multiprocessing

import numpy as np
import torch

import cv2 
from PIL import Image
import numpy as np
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import shutil

import core.utils.bvh_to_joint as btoj
from core.visualize.preprocess import preprocess


def save_timelapse(preview_path, preview_list, config):

    view_range = config.preview.rangemax
    save_format = os.path.splitext(preview_path)[1]

    # Prepare skelton
    standard_bvh = config.dataset.standard_bvh
    skelton, non_end_bones, joints_to_index, permute_xyz_order = btoj.get_standard_format(standard_bvh)
    #_, non_zero_joint_to_index = btoj.cut_zero_length_bone(skelton, joints_to_index)
    parts = btoj.divide_skelton_into_parts(skelton, joints_to_index)

    # Preprocessing data
    preview_list = preprocess(preview_list, config)


    # Define output directory
    preview_dir_top, motion_name = os.path.split(preview_path)
    motion_id = os.path.splitext(motion_name)[0]
    if len(motion_id.split('_')) > 1:
        motion_id, mix_ratio = motion_id.split('_')
    else:
        mix_ratio = ""

    output_dir = os.path.join(preview_dir_top, 'timelapse', motion_id)
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir))

    frames = []
    frame_length = preview_list[0]['motion'].shape[0]

    end = time.time()

    """ Plot configs
    """
    #'''
    # Setting 1 (Fig5)
    azim=90
    lw=6
    interval=8
    max_frames=16
    view_range=80
    elev=30
    #'''
    '''
    # Setting 2 (Compare)
    azim=0
    lw=6
    interval=4
    max_frames=16
    view_range=80
    elev=0
    '''
    '''
    # Setting 3 (Compare spline_x4)
    axim=0
    lw=6
    interval=6
    max_frames=32
    view_range=160
    elev=0
    '''
    '''
    # Setting 4 (Fig1)
    azim=90
    lw=6
    interval=8
    max_frames=16
    view_range=80
    elev=30
    '''
    '''
    # Setting 5 (Alltogether)
    azim=90
    lw=6
    interval=8
    max_frames=frame_length
    view_range=200
    elev=30
    '''
    '''
    # Setting 6 (Iwan Compare)
    azim=0
    lw=6
    interval=2
    max_frames=24
    view_range=300
    elev=0
    '''

    try:
        for data in preview_list:
            plot_figure_lapse(output_dir, data, motion_id, parts, view_range, lw, elev, azim, interval, max_frames, '_'+mix_ratio, save_format)


    except KeyboardInterrupt:
        shutil.rmtree(dirpath)
        print('\033[91m\nUser Keyboard Interrupt\033[0m')
        sys.exit()

    #shutil.rmtree(dirpath)
    print(f'Preview Saved. (Time:{time.time()-end:.04f})')
    return



def plot_figure_lapse(output_dir, data, motion_id, parts, view_range, lw, elev, azim, interval, max_frames, mix_ratio, save_format):
    motion = data['motion']
    name = data['name']
    control = data['control']

    motion = motion[::interval,:]
    frames = []
    frame_length = motion.shape[0]

    if motion.shape[1] != 84:
        # CMU Data
        head = 19
        one_stroke = [0,13,14,#15,
                        15,29,30,31,#32,
                           32,36,37,36,#32,
                           32,33,34,35,34,33,#32,
                        32,31,30,29,#15,
                        15,20,21,22,#23,
                           23,27,28,27,#23,
                           23,24,25,26,25,24,#23,
                        23,22,21,20,#15
                        15,16,17,18,19,18,17,16,#15,
                      15,14,13,#0,
                      0,7,8,9,10,11,12,11,10,9,8,7,#0
                      0,1,2,3,4,5,6]

    else:
        # Iwan Data
        motion /= view_range / 40
        if control is not None: control /= view_range / 40
        view_range = 80

        head = 7
        one_stroke = [0,23,24,25,26,#27,     Root -> LeftToe
                        27,26,25,24,23,#0,      <---
                      0,18,19,20,21,#22,     Root -> RightToe
                        22,21,20,19,18,#0,      <---
                      0,1,2,3,#4,            Root -> Chest4
                          4,13,14,15,16,#17,   Chest4 -> LeftWrist
                            17,16,15,14,13,#4,      <---
                          4,8,9,10,11,#12,     Chest4 -> RightWrist
                            12,11,10,9,8,#4,        <---
                          4,5,6,7]#            Chest4 -> Head

    # Plot each positions in positions_list
    for k in range(max(1, frame_length//max_frames)):
        trajectory = motion[k*max_frames:(k+1)*max_frames,0:3]
        camera_pos = np.average(trajectory, axis=0)

        fig = plt.figure(figsize=(30.0*view_range/80,30.0*view_range/80))
        ax = Axes3D(fig)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for key, val in ax.spines.items():
            ax.spines[key].set_color("none")
        ax.w_xaxis.set_pane_color((1.,1.,1.,1.))
        ax.w_xaxis.line.set_color("white")
        ax.w_yaxis.set_pane_color((1.,1.,1.,1.))
        ax.w_yaxis.line.set_color("white")
        ax.w_zaxis.set_pane_color((0.2,0.2,0.2,0.2))
        ax.w_zaxis.line.set_color("white")
        #ax.w_zaxis.set_pane_color((1.,1.,1.,1.))
        #ax.w_zaxis.line.set_color("white")

        if control is not None:
            st = interval*(k*max_frames)
            control_plot = control[st:st+interval*max_frames+1,:] - control[st,:] + trajectory[0,:]
            ax.plot(control_plot[:,0], control_plot[:,2], control_plot[:,1], '-', color='red', lw=lw+2)


        for i in range(min(frame_length, max_frames)):
            color = [c*0.85 for c in cm.jet((max_frames-i)/(max_frames*1.6))]
            color[0] += 0.15
            ax.set_xlim(-view_range//2+camera_pos[0], view_range//2+camera_pos[0])
            ax.set_ylim(-view_range//2+camera_pos[2], view_range//2+camera_pos[2])
            ax.set_zlim(-view_range//2+camera_pos[1], view_range//2+camera_pos[1])

            pose = motion[k*max_frames+i,:]

            root_pos = pose[:3]
            x = [pose[index*3] + root_pos[0] if index!=0 else pose[index*3] for index in one_stroke]
            y = [pose[index*3+2] + root_pos[2] if index!=0 else pose[index*3+2] for index in one_stroke]
            z = [pose[index*3+1] + root_pos[1]if index!=0 else pose[index*3+1]for index in one_stroke]
            # Head
            ax.plot([pose[head*3]+root_pos[0]], [pose[head*3+2]+root_pos[2]], [pose[head*3+1]+root_pos[1]], "o", ms=54, color=color, alpha=alpha)
            ax.plot(x, y, z, "-", ms=4, mew=0.5, linewidth=lw, color=color, alpha=alpha)


        plt.title(name)
        plt.savefig(os.path.join(output_dir, f'{name}_{motion_id}_{k:02d}{mix_ratio}{save_format}'))
        plt.clf()
        plt.cla()
        ax.clear()
        plt.close()

    return


