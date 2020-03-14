### Script to convert matlab structure file (/motiongan/data/style-dataset/style_motion_database.mat')
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import math
import numpy as np
from collections import OrderedDict
import scipy.io
import pickle

from core.utils.euler_to_quaternion import quaternion_to_rotation_mat, rotation_mat_to_euler

## Load motion data from .mat file
def load_motion(mat_path, out):
    mat_data = scipy.io.loadmat(mat_path)['motion_database']
    file_nums = mat_data.shape[1]
    motion_data_all = {}
    for f_id in range(file_nums):
        motion_data = {}
        # Get style and motion content 
        motion_data['style'] = mat_data[0,f_id][0][0]
        motion_data['motion_type'] = mat_data[0,f_id][1][0]  

        # Get file name
        full_path = mat_data[0,f_id][2][0,0][0][0]
        file_name = full_path.split('\\')[-1]

        # Get joint parameters
        frame_nums = mat_data[0,f_id][2].shape[1]
        root_pos = np.zeros((frame_nums,3))
  
        joint_nums = mat_data[0,f_id][2][0,0][2].shape[0]
        motion_data['joint_nums'] = joint_nums
        joint_quarternions = np.zeros((frame_nums, joint_nums, 4))
        for i in range(frame_nums):
            root_pos[i,:] = mat_data[0,f_id][2][0,i][1]
            joint_quarternions[i,:,:] = mat_data[0,f_id][2][0,i][2]
        motion_data['root_position'] = root_pos
        motion_data['joint_quarternions'] = joint_quarternions

        # Get foot contact annotation
        motion_data['foot_contact'] = mat_data[0,f_id][3][0]


        # Save file as pickle
        with open(os.path.join(out, os.path.splitext(file_name)[0]+'.pkl'), 'wb') as f:
            pickle.dump(motion_data, f)

        motion_data_all[file_name] = motion_data

    return motion_data_all


## Load skeleton data from .mat file
def load_skeleton(mat_path):
    mat_data = scipy.io.loadmat(mat_path)['skel'][0,0]

    # Init skeleton
    skeleton = OrderedDict()
    bone_names = mat_data[1].tolist()
    for i, bone in enumerate(bone_names):
        bone = bone.strip()
        if bone == 'Site':
            bone = bone_names[i-1].strip() + bone
        skeleton[bone] = {'offset':[], 'parent':[], 'children':[]}
        
    # Resister bone parent and children, offset
    parent_ids = mat_data[2][0]
    offsets = mat_data[3]
    for i, bone in enumerate(skeleton.keys()):
        if bone != 'root': 
            parent = list(skeleton.keys())[parent_ids[i]-1]
            skeleton[bone]['parent'] = parent
            skeleton[parent]['children'].append(bone)

        skeleton[bone]['offset'] = offsets[i,:]

    return skeleton


## Construct hierarchy of skeleton for bvh
def construct_hierarchy(skeleton):
    hierarchy = ['HIERARCHY\r\n']
    
    # Calc tree level
    level = 0
    for i, bone in enumerate(skeleton.keys()):
        if bone == 'root':
            skeleton[bone]['level'] = 0
        else:
            parent = skeleton[bone]['parent']
            skeleton[bone]['level'] = skeleton[parent]['level'] + 1

    # Write hierarchy
    for i, bone in enumerate(skeleton.keys()):
        offset = skeleton[bone]['offset']
        if bone == 'root':
            hierarchy.append('ROOT root\r\n')
            hierarchy.append('{\r\n')
            hierarchy.append('\tOFFSET {0:.05f} {1:.05f} {2:.05f}\r\n'.format(offset[0],offset[1],offset[2]))
            hierarchy.append('\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\r\n')

        elif bone.endswith('Site'):
            parent = skeleton[bone]['parent']
            level = skeleton[bone]['level']
            tabs = '\t' * level
            hierarchy.append(tabs + 'End Site\r\n')
            hierarchy.append(tabs + '{\r\n')
            hierarchy.append(tabs + '\tOFFSET {0:.05f} {1:.05f} {2:.05f}\r\n'.format(offset[0],offset[1],offset[2]))
            hierarchy.append(tabs + '}\r\n')
            # Put end brancket
            if i == len(skeleton.keys())-1:
                while level > 0:
                    level -= 1
                    hierarchy.append('\t' * level + '}\r\n')
            else: 
                for _ in range(level - skeleton[list(skeleton.keys())[i+1]]['level']):
                    level -= 1
                    hierarchy.append('\t' * level + '}\r\n')

        else:
            parent = skeleton[bone]['parent']
            level = skeleton[bone]['level']
            tabs = '\t'*level
            hierarchy.append(tabs + 'JOINT {0}'.format(bone) + '\r\n')
            hierarchy.append(tabs + '{\r\n')
            hierarchy.append(tabs + '\tOFFSET {0:.05f} {1:.05f} {2:.05f}\r\n'.format(offset[0],offset[1],offset[2]))
            hierarchy.append(tabs + '\tCHANNELS 3 Zrotation Yrotation Xrotation\r\n')
        
    #with open('hierarchy_test.txt', 'w') as f:
    #    f.writelines(hierarchy)
    return hierarchy


# Write .bvh file
def write_bvh(skeleton, hierarchy, motion_data_all, out):
    for file_name, motion_data in motion_data_all.items():
        joint_quarternions = motion_data['joint_quarternions']
        root_pos = motion_data['root_position']

        # Convert data to list of string
        frames = []
        for i in range(joint_quarternions.shape[0]):
            # Root pos
            root_pos_i = root_pos[i]
            frame = '{0:.05f} {1:.05f} {2:.05f} '.format(*root_pos_i.tolist()) 

            for j in range(joint_quarternions.shape[1]):
                # If Endsite, skip
                if list(skeleton.keys())[j].endswith('Site'): 
                    continue
                ## This implementation is modified to quarternion with 'xyzw' order
                R_ij = quaternion_to_rotation_mat(joint_quarternions[i,j,3], joint_quarternions[i,j,2], joint_quarternions[i,j,1], joint_quarternions[i,j,0])    
                euler_ij = rotation_mat_to_euler(R_ij)
                frame += '{0:.05f} {1:.05f} {2:.05f} '.format(*list(map(lambda s: s * (180.0/math.pi), euler_ij.tolist())))

            frame += '\r\n'
            frames.append(frame)
   
        # Write
        with open(os.path.join(out, file_name), 'w') as f:
            f.writelines(hierarchy)

            f.write('MOTION\r\n')
            frames[0] = 'Frames: {0}\r\nFrame Time: 0.0083333\r\n'.format(joint_quarternions.shape[0]) + frames[0]
            f.writelines(frames)
        
        print(os.path.join(out, file_name))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=str)

    args = parser.parse_args()
    out = args.out

    motion_data_all = load_motion('../../motiongan/data/style-dataset/style_motion_database.mat', out)
    skeleton = load_skeleton('../../motiongan/data/style-dataset/skeleton.mat')
    hierarchy = construct_hierarchy(skeleton)
    write_bvh(skeleton, hierarchy, motion_data_all, out) 

if __name__ == '__main__':
    main()
