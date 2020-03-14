import os
import sys

import numpy as np

import core.utils.euler_to_xyz as helper
from core.utils.bvh_to_joint import get_standard_format
from core.utils.read_bvh_hierarchy import read_bvh_hierarchy



# Get frame format (frame nums, format) from bvh
def get_frame_format_string(bvh_filename):
    with open(bvh_filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('MOTION'):
            format_lines = lines[i:i+3]
            break
    return format_lines    


# Get hashmap between joint and global position  
def data_vec_to_position_dic(data, skelton, joints_to_index):
    data[3:] = data[3:] * 100
    root_pos = data[joints_to_index['Root']*3:joints_to_index['Root']*3+3]
    positions_dic = {}
    for joint in joints_to_index:
        positions_dic[joint] = data[joints_to_index[joint]*3:joints_to_index[joint]*3+3]
    for joint in positions_dic.keys():
        if joint!='Root':
            positions_dic[joint] += positions_dic['Root']
    return positions_dic


# Convert xyz position to list of 1d vector
def xyz_to_1dvector(xyz_motion, skelton, non_end_bones):
    bvh_vec_length = len(non_end_bones)*3+6
    
    out_data = np.zeros((len(xyz_motion), bvh_vec_length)) 
    for i in range(len(xyz_motion)):
        position = xyz_motion[i]
        rotation_matrices, rotation_angles = helper.pos_to_rotation_mat(skelton, position, non_end_bones)
        motion_vec = helper.rotation_dic_to_vec(rotation_angles, non_end_bones, position)
        out_data[i,:] = np.array(motion_vec)

    
    return out_data
       

# Centering root position
def centering_root(data):
    center = np.average(data[:,0:3], axis=0)
    data[:,0:3] = data[:,0:3] - center 
    return data
    
# Write frame data
def write_data_to_bvh(format_data, out_bvh, standard_bvh):
    # Get standard skelton
    skelton, non_end_bones, joints_to_index, _ = get_standard_format(standard_bvh)
    # Get frame data to proper format
    seq_length = format_data.shape[0]
    xyz_motion = []
    for i in range(seq_length):
        data = np.array(format_data[i,:])
        position_dic = data_vec_to_position_dic(data, skelton, joints_to_index)
        xyz_motion.append(position_dic)
    out_data = xyz_to_1dvector(xyz_motion, skelton, non_end_bones)

    # Centering
    out_data= centering_root(out_data)

    # Get hierarchy text
    hierarchy_text = ''
    with open(standard_bvh, 'r') as f_st:
        for line in f_st:
            if line.startswith('MOTION'):
                break
            hierarchy_text += line

    # Get frame format
    frame_format =  get_frame_format_string(standard_bvh)
    frame_nums = len(out_data)
    frame_format[1] = f'Frames:\t{frame_nums}\n'

    # Write 
    with open(out_bvh, 'w') as f_out:
        f_out.write(hierarchy_text)
        f_out.writelines(frame_format)
        f_out.writelines([' '.join(list(map(lambda d:str(round(d,6)), data))) + '\n' for data in out_data])
    

def debug():
    format_data = np.load('results/TimeTRUNet/TimeTRUNet_Walking_advonly_styleGAN_startloss_lowpass/test/iter_90000/Walking_jp/split/002_real.npy')
    write_data_to_bvh(format_data, 'test.bvh', 'core/utils/CMU_standard.bvh')

    
