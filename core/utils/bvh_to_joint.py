import os
import sys
import math
import subprocess

import cv2
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import core.utils.euler_to_xyz as euler_to_xyz
from core.utils.read_bvh_hierarchy import read_bvh_hierarchy


# Get index of each joint in motion part
def get_pos_joints_index(frame, skelton, non_end_bones, permute_xyz_order):
    pos_dic = euler_to_xyz.get_pos_skelton(frame, skelton, non_end_bones, permute_xyz_order)
    joints_to_index = OrderedDict()
    for i, joint in enumerate(list(pos_dic.keys())):
        joints_to_index[joint] = i
    return joints_to_index


# Parse frame data from bvh file
def parse_frames(bvh_path, permute_xyz_order):
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
    # Find start line of motion part
    for i, l in enumerate(lines):
        if l.find('MOTION') > -1:
            first_frame = i + 3
            break

    num_params = len(lines[first_frame].split())
    num_frames = len(lines) - first_frame
    motion = np.zeros((num_frames, num_params))

    for t in range(num_frames):
        line = lines[first_frame+t].split()
        for joint in range(1,len(line)//3):
            line[joint*3:joint*3+3] = [line[joint*3+p] for p in permute_xyz_order]
        motion[t,:] = np.array(list(map(lambda e : float(e), line)))

    return motion


# Get joint positions from euler angle of each frame
def get_one_frame_format_data(frame, skelton, non_end_bones, permute_xyz_order):
    pos_dic = euler_to_xyz.get_pos_skelton(frame, skelton, non_end_bones, permute_xyz_order)
    format_data = np.zeros(len(list(pos_dic.keys()))*3)
    
    root_pos = pos_dic['Root'].reshape(3)
    # Set each joint position
    for i, joint in enumerate(list(pos_dic.keys())):
        if i == 0:
            format_data[i*3:i*3+3] = root_pos
        else:
            # Relative position from root
            format_data[i*3:i*3+3] = pos_dic[joint].reshape(3) - root_pos
    return format_data

        
# Get joint positions of all frame
def get_format_motion(raw_frames_data, skelton, non_end_bones, permute_xyz_order):
    format_motion = []
    for i, frame in enumerate(raw_frames_data):
        format_motion.append(get_one_frame_format_data(frame, skelton, non_end_bones, permute_xyz_order))
    format_motion = np.array(format_motion)
        
    #format_motion = np.array([get_one_frame_format_data(frame, skelton, non_end_bones) for frame in raw_frames_data])
    return format_motion


# Cut bone whose length is 0
def cut_zero_length_bone_frames(format_data, skelton, joints_to_index):
    non_zero_bones, non_zero_joint_to_index = cut_zero_length_bone(skelton, joints_to_index)

    non_zero_format_data = np.zeros((format_data.shape[0], len(non_zero_bones)*3)) 
    for nz_i, joint in enumerate(non_zero_bones):
        i = joints_to_index[joint]
        positions = format_data[:,i*3:i*3+3]
        non_zero_format_data[:,nz_i*3:nz_i*3+3] = positions
    return non_zero_joint_to_index, non_zero_format_data 

def cut_zero_length_bone(skelton, joints_to_index, threshold=1e-8):
    non_zero_bones = ['Root']
    for joint in skelton.keys():
        offsets = skelton[joint]['offsets']
        offsets_norm = offsets[0]**2+offsets[1]**2+offsets[2]**2
        if offsets_norm > threshold:
            non_zero_bones.append(joint)
    
    non_zero_joint_to_index = {}
    for nz_i, joint in enumerate(non_zero_bones):
        non_zero_joint_to_index[joint] = nz_i

    return non_zero_bones, non_zero_joint_to_index


# Put back bone whose length is 0
def put_zero_length_bone(non_zero_format_data, non_zero_joint_to_index, skelton, joints_to_index, threshold=1e-8):
    format_data = np.zeros((non_zero_format_data.shape[0], len(list(skelton.keys()))*3))
    for joint in skelton.keys():
        if joint in non_zero_joint_to_index:
            i = joints_to_index[joint]
            nz_i = non_zero_joint_to_index[joint] 
        else:
            parent = skelton[joint]['parent']
            while parent not in non_zero_joint_to_index and parent != 'Root':
                parent = skelton[parent]['parent']
            if parent == 'Root':
                continue
            i = joints_to_index[joint]
            nz_i = non_zero_joint_to_index[parent]
        format_data[:,i*3:i*3+3] = non_zero_format_data[:,nz_i*3:nz_i*3+3]
    return format_data



# Divide skelton into parts for visualization
def divide_skelton_into_parts(skelton, joints_to_index):
    parts = []
    root, root_bone = list(skelton.items())[0]
    joint_stack = [(root, [])]
    while joint_stack:
        current, current_part = joint_stack.pop()
        current_index = joints_to_index[current]
        current_part.append((current_index, current))

        children = skelton[current]['children']
        if len(children) == 1:
            joint_stack.append((children[0], current_part))
        elif len(children) > 1:
            if len(current_part) > 1:
                parts.append(current_part)
            for child in children:
                joint_stack.append((child, [(current_index, current)]))
        else:
            # End site
            if len(current_part) > 1:
                parts.append(current_part)
    return parts

    
# Collect all bone(here, not bone object, just collect index pair)
def collect_bones(bvh_path):
    skelton, _, joints_to_index, _ = get_standard_format(bvh_path)
    _, non_zero_joint_to_index = cut_zero_length_bone(skelton, joints_to_index)

    # collect zero joint
    for joint in joints_to_index:
        if joint not in non_zero_joint_to_index:
            parent = skelton[joint]['parent']
            non_zero_joint_to_index[joint] = non_zero_joint_to_index[parent]
  
    bones = []
    root, _ = list(skelton.items())[0]
    joint_stack = [root]
    while joint_stack:
        current = joint_stack.pop()
        current_index = non_zero_joint_to_index[current]

        children = skelton[current]['children']
        if children:
            for child in children:
                child_index = non_zero_joint_to_index[child]
                if current_index != child_index:
                    bones.append((current_index, child_index))
                joint_stack.append(child)
    return bones


# Get skelton format
def get_standard_format(standard_bvh):
    skelton, non_end_bones, permute_xyz_order = read_bvh_hierarchy(standard_bvh)
    #print(skelton)
    raw_frames_data = parse_frames(standard_bvh, permute_xyz_order)
    joints_to_index = get_pos_joints_index(raw_frames_data[0], skelton, non_end_bones, permute_xyz_order)
    return skelton, non_end_bones, joints_to_index, permute_xyz_order


# main
def create_data(bvh_path, standard_skelton, joints_to_index, max_frames_num=-1):
    # Parse hierarchy
    skelton, non_end_bones, joints_to_index, permute_xyz_order = get_standard_format(bvh_path)
    # Parse frame data
    raw_frames_data = parse_frames(bvh_path, permute_xyz_order)
    if max_frames_num > 0: raw_frames_data = raw_frames_data[:max_frames_num]
    
    # Use standard skelton
    assert standard_skelton.keys() == skelton.keys()

    format_motion = get_format_motion(raw_frames_data, standard_skelton, non_end_bones, permute_xyz_order)
    return format_motion



# For debug
def show(positions, skelton, joints_to_index, save_name): 
    parts = divide_skelton_into_parts(skelton, joints_to_index)
    center = np.average(positions[:,0:3], axis=0)
    for t in range(positions.shape[0]):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)
        pose = positions[t,:]
        for part in parts:
            x, y, z = np.zeros(len(part)), np.zeros(len(part)), np.zeros(len(part))
            for i, (index, name) in enumerate(part):
                if name == 'Root':
                    x[i] = pose[index*3] - center[0]
                    z[i] = pose[index*3+1] - center[1]
                    y[i] = pose[index*3+2] - center[2]
                    root_pos = pose[index*3:index*3+3] - center
                else:
                    x[i] = pose[index*3] + root_pos[0]
                    z[i] = pose[index*3+1] + root_pos[1]
                    y[i] = pose[index*3+2] + root_pos[2]
                #ax.text(x[i], y[i], z[i], name, size=6)
            ax.plot(x, y, z, "-o", ms=4, mew=0.5, linewidth=5)
        
        plt.savefig(f'test_convert/{save_name}_{t:05d}.png')
        plt.close()

    cmd = ['convert','-layers','optimize','-loop','0','-delay', '0',f'test_convert/{save_name}_*.png',f'test_convert/{save_name}.gif']
    subprocess.run(cmd)


def debug():
    bvh_paths = ['../motiongan/data/train_jp/Edi_Mocap_Data/Labelled/WalkRandom.bvh'] 
    skelton, non_end_bones, joints_to_index, permute_xyz_order = get_standard_format(bvh_paths[0])
    collect_bones(skelton, joints_to_index)
    sys.exit()
    for bvh_path in bvh_paths:
        _, non_zero_joint_to_index = cut_zero_length_bone(skelton, joints_to_index)

        format_data = create_data(bvh_path, skelton, joints_to_index)
        format_data = format_data[:1000,:]
        non_zero_joint_to_index, non_zero_format_data = cut_zero_length_bone_frames(format_data, skelton, joints_to_index)
        format_data = put_zero_length_bone(non_zero_format_data, non_zero_joint_to_index, skelton, joints_to_index)
        name = os.path.splitext(os.path.split(bvh_path)[1])[0]
        show(format_data[::4,:], skelton, joints_to_index, name)

if __name__ == '__main__':
    debug() 
