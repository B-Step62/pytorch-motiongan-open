import os
import sys
import math
import shutil
import pickle
import argparse

import numpy as np
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.utils import bvh_to_joint as converter


# Copy all original bvh file
def collect_bvh(bvh_root, out):
    root_depth = bvh_root.count(os.path.sep)
    bvh_paths = []
    out_dir = out
    for (root, dirs, files) in os.walk(bvh_root):
        for origin_file in files:
            if not origin_file.endswith('.bvh'):
                continue
            # Output path is 'out' + ('origin_path' - 'root')
            if bvh_root != out:
                post = root.split(os.path.sep)[root_depth:]
                out_dir = out + ''.join([os.path.sep + p for p in post])
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                # If save to different directory, also copy original bvh
                shutil.copy(os.path.join(root, origin_file), os.path.join(out_dir, origin_file))
                bvh_paths.append(os.path.join(out_dir, origin_file))
            else:
                bvh_paths.append(os.path.join(root, origin_file))
        
    return out_dir, bvh_paths


# Convert bvh to joint_position
def convert_bvh(bvh_paths, out_dir, standard_bvh=None):        

    # Parse standard hierarchy
    standard_skelton, non_end_bones, joints_to_index, permute_xyz_order = converter.get_standard_format(standard_bvh)
    
    for bvh_path in bvh_paths:
        out_npy = os.path.splitext(bvh_path)[0] + '.npy'      
        # Parse hierarchy
        skelton,  _, _, permute_xyz_order_b = converter.get_standard_format(bvh_path)
        assert permute_xyz_order == permute_xyz_order_b
        if standard_skelton.keys() != skelton.keys():
            print('Skelton mismatch with standard bvh!')
            sys.exit()
        # Parse frame data
        raw_frames_data = converter.parse_frames(bvh_path, permute_xyz_order)

        # Save format motion
        format_motion = converter.get_format_motion(raw_frames_data, standard_skelton, non_end_bones, permute_xyz_order)
        out_npy = os.path.splitext(bvh_path)[0] + '.npy'      
        print(f'Generate {out_npy} [Shape:{format_motion.shape}]')
        np.save(out_npy, format_motion) 
