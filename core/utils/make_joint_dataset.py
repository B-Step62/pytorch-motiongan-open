import os
import sys 
import math
import subprocess

import cv2 
from collections import OrderedDict
import numpy as np

import core.utils.bvh_to_joint as btoj


BVH_ROOT = './data/bvh/Edi_Mocap_Data/Iwan_style_data'
OUT = './data/bvh/Edi_Mocap_Data/Iwan_style_data'


def main():
    # Copy all original bvh file
    root_depth = BVH_ROOT.count(os.path.sep)
    bvh_paths = []
    out_dir = OUT
    for (root, dirs, files) in os.walk(BVH_ROOT):
        for origin_file in files:
            if not origin_file.endswith('.bvh'):
                continue
            # Output path is 'out' + ('origin_path' - 'root')
            if BVH_ROOT != OUT:
                post = root.split(os.path.sep)[root_depth:]
                out_dir = OUT + ''.join([os.path.sep + p for p in post])
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                # If save to different directory, copy original bvh
                shutil.copy(os.path.join(root, origin_file), os.path.join(out_dir, origin_file))
                bvh_paths.append(os.path.join(out_dir, origin_file))
            else:
                bvh_paths.append(os.path.join(root, origin_file))
        
    
    skelton, non_end_bones, joints_to_index, permute_xyz_order = btoj.get_standard_format(bvh_paths[0])
    
    for bvh_path in bvh_paths:
        _, non_zero_joint_to_index = btoj.cut_zero_length_bone(skelton, joints_to_index)
    
        format_data = btoj.create_data(bvh_path, skelton, joints_to_index)
        npy_path = os.path.splitext(bvh_path)[0] + '.npy'
        np.save(npy_path, format_data)
        print(npy_path, format_data.shape)
