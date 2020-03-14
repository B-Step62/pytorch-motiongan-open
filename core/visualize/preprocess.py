#!/usr/bin/env python
import os

import numpy as np
import torch

import core.utils.bvh_to_joint as btoj


def preprocess(preview_list, config):

    # Laod paramters
    scale = config.dataset.scale
    standard_bvh = config.dataset.standard_bvh


    # Prepare skelton
    skelton, non_end_bones, joints_to_index, permute_xyz_order = btoj.get_standard_format(standard_bvh)
    _, non_zero_joint_to_index = btoj.cut_zero_length_bone(skelton, joints_to_index)


    # Processing data 
    for preview in preview_list:

        motion = preview['motion']
        if isinstance(motion, torch.FloatTensor):
            motion = motion[0,:,:].numpy()
        if len(motion.shape) == 3:
            motion = motion[0,:,:]

        # Back to original scale
        motion *= scale
        motion = btoj.put_zero_length_bone(motion, non_zero_joint_to_index, skelton, joints_to_index) 
        
        preview['motion'] = motion

        control = preview['control']
        if control is not None: 
            if isinstance(control, torch.FloatTensor):
                control = control[0,:,:].data.cpu().numpy()
            if len(control.shape) == 3:
                control = control[0,:,:]
            # Back to original scale as well as motion
            control = control * scale
        preview['control'] = control


    return preview_list
