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


def save_video(preview_path, preview_list, config, camera_move='chase', elev=15, azim=45, lw=5):

    if not os.path.exists(os.path.split(preview_path)[0]):
        os.makedirs(os.path.split(preview_path)[0])

    view_range = config.preview.view_range
    save_delay = config.preview.save_delay
    save_format = os.path.splitext(preview_path)[1]

    preview_list = preprocess(preview_list, config)

    # Parse skelton and divide it into several body parts
    standard_bvh = config.dataset.standard_bvh
    skelton, non_end_bones, joints_to_index, permute_xyz_order = btoj.get_standard_format(standard_bvh)
    parts = btoj.divide_skelton_into_parts(skelton, joints_to_index)

    # Create temporal file save directory
    i = 0
    tmp_out_dir = os.path.join('tmp', str(i))
    while os.path.exists(tmp_out_dir):
        tmp_out_dir = os.path.join('tmp', str(i))
        i += 1
    os.makedirs(tmp_out_dir)


    # Save each preview frame
    try:
        frames = []
        frame_length = preview_list[0]['motion'].shape[0]

        end = time.time()

        jobs = []
        for j, data in enumerate(preview_list):
            p = multiprocessing.Process(target=plot_figure, args=(tmp_out_dir, data, j, parts, view_range, lw, elev, azim, camera_move))
            jobs.append(p)
            p.start()

        for pj in jobs:
            pj.join()

        # Save image of each frame in tmp directory
        WIDTH = 1080
        HEIGHT = 810
        COLUMN = min(len(preview_list), 4)
        row = math.ceil(len(preview_list)/COLUMN)
        image_size = (WIDTH//row, HEIGHT//row)

        if save_format == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            writer = cv2.VideoWriter(preview_path, fourcc, 120.0/save_delay, (image_size[0]*COLUMN, image_size[1]*row))

        # Concat each frame
        for t in range(frame_length):
            for i in range(row):
                for j in range(4):
                    if j == 0:
                        im_rowcol = cv2.resize(cv2.imread(f'{tmp_out_dir}/{t:05d}_{i*4+j:02d}.png'), image_size)
                        im_row = im_rowcol
                    elif i*4+j >= len(preview_list):
                        if row == 1:
                            break
                        im_rowcol = cv2.resize(np.zeros_like(im_rowcol).astype(np.uint8), image_size)
                        im_row = cv2.hconcat([im_row, im_rowcol])
                    else:
                        im_rowcol = cv2.resize(cv2.imread(f'{tmp_out_dir}/{t:05d}_{i*4+j:02d}.png'), image_size)
                        im_row = cv2.hconcat([im_row, im_rowcol])
                if i == 0:
                    im = im_row
                else:
                    im = cv2.vconcat([im, im_row])

            if save_format == '.avi':
                writer.write(im)
                sys.stdout.write(f'\rvideo writing... {t}/{frame_length}')
                sys.stdout.flush()
            elif save_format == '.gif':
                cv2.imwrite(f'{tmp_out_dir}/concat_{t:05d}.png', im)


        if save_format == '.avi':
            cv2.destroyAllWindows()
            writer.release()
        elif save_format == '.gif':
            # Combine frame images into gif by imagemagick convert
            cmd = ['convert','-layers','optimize','-loop','0','-delay', f'{save_delay}',f'{tmp_out_dir}/concat_*.png',f'{preview_path}']
            subprocess.run(cmd)

    except KeyboardInterrupt:
        shutil.rmtree(tmp_out_dir)
        print('\033[91m\nUser Keyboard Interrupt\033[0m')
        sys.exit()

    shutil.rmtree(tmp_out_dir)
    print(f'Preview Saved. (Time:{time.time()-end:.04f})')

    return


def plot_figure(tmp_out_dir, data, motion_id, parts, view_range, lw, elev, azim, camera_mode):
    caption = data['caption']
    motion = data['motion']
    control = data['control']
 
    frames = []
    frame_length = motion.shape[0]


    # Plot each frame in motion
    trajectory = motion[:,0:3]
    for frame_id in range(frame_length):
        # Plot initialization
        fig = plt.figure()
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
        ax.w_zaxis.set_pane_color((1.,1.,1.,1.))
        ax.w_zaxis.line.set_color("white")

        # Camera mode : fixed or not
        if camera_mode == 'chase':
            camera_pos = np.average(trajectory[max(0,frame_id-2):min(frame_id+2,trajectory.shape[0]),:], axis=0)
        elif camera_mode == 'stand':
            camera_pos = (np.max(trajectory, axis=0) + np.min(trajectory, axis=0)) / 2

        # Initialize view range
        ax.set_xlim(-view_range//2+camera_pos[0], view_range//2+camera_pos[0])
        ax.set_ylim(-view_range//2+camera_pos[2], view_range//2+camera_pos[2])
        ax.set_zlim(-view_range//2+camera_pos[1], view_range//2+camera_pos[1])

        # Draw character
        pose = motion[frame_id,:]
        for part in parts:
            # Resister each body parts separately
            x, y, z = np.zeros(len(part)), np.zeros(len(part)), np.zeros(len(part))
            for p, (index, joint_name) in enumerate(part):
                if joint_name == 'Root':
                    x[p] = pose[index*3]
                    z[p] = pose[index*3+1]
                    y[p] = pose[index*3+2]
                    root_pos = pose[index*3:index*3+3]
                elif joint_name in ['LeftToeBase_EndSite', 'RightToeBase_EndSite']:
                    x = x[:-1]
                    y = y[:-1]
                    z = z[:-1]
                else:
                    x[p] = pose[index*3] + root_pos[0]
                    z[p] = pose[index*3+1] + root_pos[1]
                    y[p] = pose[index*3+2] + root_pos[2]
                if joint_name == 'Head_EndSite':
                   # Draw head
                   u, v = np.mgrid[0:np.pi:20j, 0:np.pi:10j]
                   ax.plot_wireframe(2*np.cos(u)*np.sin(v)+x[p], 2*np.sin(u)*np.sin(v)+y[p], 2*np.cos(v)+z[p], color='black', alpha=0.4)
                   ax.plot_wireframe(2*np.cos(u)*np.sin(-v)+x[p], 2*np.sin(u)*np.sin(-v)+y[p], 2*np.cos(v)+z[p], color='black', alpha=0.4)
            # Plot resistered body parts
            ax.plot(x, y, z, "-o", ms=4, mew=0.5, linewidth=4, color=[0.3,0.3,0.3], alpha=0.8)
            for xp, yp, zp in zip(x,y,z):
                ax.plot([xp], [yp], [zp], ".", ms=7, mew=0.5, linewidth=4, color='black')


        # Draw control curve
        if control is not None:
            st, en = max(0, frame_id-32), min(frame_id+32, control.shape[0])
            ax.plot(control[st:en-2,0], control[st:en-2,2], control[st:en-2,1], lw=lw-2, color='red', alpha=0.8)
            ax.plot([control[en-2,0]], [control[en-2,2]], [control[en-2,1]], "o", ms=8, mew=0.5, linewidth=4, color='red', alpha=0.8)


        plt.title(caption)
        plt.savefig(f'{tmp_out_dir}/{frame_id:05d}_{motion_id:02d}.png')


        plt.clf()
        plt.cla()
        ax.clear()
        plt.close()

    return
