import os
import math
import re

from collections import OrderedDict
import numpy as np
import transforms3d.euler as euler
from scipy.spatial.transform import Rotation


# Alias
def sin(theta): return math.sin(theta)
def cos(theta): return math.cos(theta)


# Convert euler angles to rotation matrix
def euler_to_rotation_mat(euler, permute_xyz_order): 
    x, y, z = euler

    x = x / 180 * np.pi
    y = y / 180 * np.pi
    z = z / 180 * np.pi

    R_x = np.array([[       1,       0,        0],
                    [       0,  cos(x),  -sin(x)],
                    [       0,  sin(x),   cos(x)]])

    R_y = np.array([[  cos(y),       0,   sin(y)],
                    [       0,       1,        0],
                    [ -sin(y),       0,   cos(y)]])

    R_z = np.array([[  cos(z), -sin(z),        0],
                    [  sin(z),  cos(z),        0],
                    [       0,       0,        1]])

    order_l = [None, None, None]
    order_l[permute_xyz_order[0]] = 'z'
    order_l[permute_xyz_order[1]] = 'y'
    order_l[permute_xyz_order[2]] = 'x'
    order = ''.join(order_l) 
    if order=='zyx':  # CMU Data
        return np.dot(R_z, np.dot(R_y, R_x)) 
    elif order=='zxy':
        return np.dot(R_z, np.dot(R_x, R_z)) 
    elif order=='yzx':
        return np.dot(R_y, np.dot(R_z, R_x)) 
    elif order=='yxz': # Iwan Data
        return np.dot(R_y, np.dot(R_x, R_z)) 
    elif order=='xzy':
        return np.dot(R_x, np.dot(R_z, R_y)) 
    elif order=='xyz':
        return np.dot(R_x, np.dot(R_y, R_z)) 
    else:
        raise ValueError(f'Invalid euler-axes order : {order}.')
        return

# Convert rotation matrix to euler angles
def rotation_mat_to_euler(R):
    cosy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        print('singular')
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


# Calcurate global transformation matrix 
def get_global_transform(bone, skelton, motion, non_end_bones, permute_xyz_order):
    parent = skelton[bone]['parent']
    transform_mat = get_relative_transform(bone, skelton, motion, non_end_bones, permute_xyz_order)
    if bone == 'RightKnee':
        bone_index = non_end_bones.index(bone)
        z_euler = motion[6+3*bone_index]
        y_euler = motion[6+3*bone_index+1]
        x_euler = motion[6+3*bone_index+2]
    # Inversively get transform matrix from root
    while parent != None:
        parent_transform_mat = get_relative_transform(parent, skelton, motion, non_end_bones, permute_xyz_order)
        transform_mat = np.dot(parent_transform_mat, transform_mat)
        parent = skelton[parent]['parent']
    return transform_mat


# Calcurate root (usually 'Hips') transform matrix 
def get_root_transform(motion, skelton, permute_xyz_order):
    # Rotation matrix
    z_euler = motion[3]
    y_euler = motion[4]
    x_euler = motion[5]
    axes = ['x', 'y', 'z']
    rotation_mat = euler_to_rotation_mat([x_euler, y_euler, z_euler], permute_xyz_order)
    # Set the origin as root position
    root_offsets = np.array(motion[0:3])
    # Create affine transform matrix using root position (translation) and rotation matrix
    transform_mat = np.zeros((4,4))
    transform_mat[0:3, 0:3] = rotation_mat
    transform_mat[0, 3] = root_offsets[0]
    transform_mat[1, 3] = root_offsets[1]
    transform_mat[2, 3] = root_offsets[2]
    transform_mat[3, 3] = 1
    return transform_mat 


# Calcurate relative transformation matrix
def get_relative_transform(bone, skelton, motion, non_end_bones, permute_xyz_order): 
    # Rotation matrix
    if bone in non_end_bones:
        bone_index = non_end_bones.index(bone)
        z_euler = motion[6+3*bone_index]
        y_euler = motion[6+3*bone_index+1]
        x_euler = motion[6+3*bone_index+2]
        rotation_mat = euler_to_rotation_mat([x_euler, y_euler, z_euler], permute_xyz_order)
    else:
        rotation_mat = np.identity(3)
    # Offset of bone
    offsets = np.array(skelton[bone]['offsets'])
    # Create affine transform matrix from parent bone 
    transform_mat = np.zeros((4,4))
    transform_mat[0:3, 0:3] =  rotation_mat
    transform_mat[0, 3] = offsets[0] 
    transform_mat[1, 3] = offsets[1] 
    transform_mat[2, 3] = offsets[2] 
    transform_mat[3, 3] = 1
    return transform_mat


# Get position of each bone 
def get_pos_bone(bone, skelton, motion, non_end_bones, permute_xyz_order):
    global_transform_mat = np.dot(get_root_transform(motion, skelton, permute_xyz_order), get_global_transform(bone, skelton, motion, non_end_bones, permute_xyz_order))
    position = np.dot(global_transform_mat, np.array([[0, 0, 0, 1]]).T)
    return position

# Get position of all bone in ekelton
def get_pos_skelton(motion, skelton, non_end_bones, permute_xyz_order):
    pos_dic = OrderedDict()
    for bone in skelton.keys():
        pos = get_pos_bone(bone, skelton, motion, non_end_bones, permute_xyz_order)
        pos_dic[bone] = pos[0:3]
    return pos_dic



# Convert rotation angle dictionary to 1d-vector format
def rotation_dic_to_vec(rotation_dic, non_end_bones, position):
    motion_vec = np.zeros(6 + len(non_end_bones)*3)
    motion_vec[0:3] = position['Root']
    motion_vec[3] = rotation_dic['Root'][2]
    motion_vec[4] = rotation_dic['Root'][1]
    motion_vec[5] = rotation_dic['Root'][0]
    for i in range(len(non_end_bones)):
        motion_vec[3*(i+2)] = rotation_dic[non_end_bones[i]][2]
        motion_vec[3*(i+2) + 1] = rotation_dic[non_end_bones[i]][0]
        motion_vec[3*(i+2) + 2] = rotation_dic[non_end_bones[i]][1]
    return motion_vec

# Convert positions to rotation matrix
def pos_to_rotation_mat(skelton, position, non_end_bones):
    all_rotation = {}
    all_rotation_matrices = {}
    
    while len(non_end_bones)-1 > len(all_rotation_matrices.keys()):
        for bone_name in ['Root']+non_end_bones:
            parent = skelton[bone_name]['parent']
            if bone_name in all_rotation_matrices.keys():
                continue
            if (not parent in all_rotation_matrices.keys()) and (parent is not None):
                continue
            # Calcurate rotarion matrix from root
            parent_rot = np.identity(3)
            while parent is not None:
                parent_rot = np.dot(all_rotation_matrices[parent], parent_rot)
                parent = skelton[parent]['parent']

            children = skelton[bone_name]['children']
            children_positions = np.zeros((len(children), 3))
            children_offsets = np.zeros((len(children), 3))
            for i in range(len(children)):
                children_positions[i, :] = np.array(position[children[i]]) - np.array(position[bone_name])
                children_offsets[i, :] = np.array(skelton[children[i]]['offsets'])
                children_positions[i, :] = children_positions[i, :] * np.linalg.norm(children_offsets[i, :]) / np.linalg.norm(children_positions[i, :]+1e-32)
                assert np.allclose(np.linalg.norm(children_positions[i,:]), np.linalg.norm(children_offsets[i,:]))

            parent_space_children_positoins = np.dot(children_positions, parent_rot)
            rotation_mat = kabsch(parent_space_children_positoins, children_offsets) 
            if bone_name == 'Root':
                all_rotation[bone_name] = np.array(euler.mat2euler(rotation_mat, 'sxyz')) * (180.0/math.pi)#rotation_mat_to_euler(rotation_mat) * (180.0 / math.pi)
            else:
                angles = np.array(euler.mat2euler(rotation_mat, 'sxyz')) * (180.0/math.pi)#rotation_mat_to_euler(rotation_mat) * (180.0 / math.pi)
                all_rotation[bone_name] = [angles[1], angles[0], angles[2]]
            all_rotation_matrices[bone_name] = rotation_mat

    return all_rotation_matrices, all_rotation


# Get rotation matrix by kabsch algorithm
def kabsch(p, q):
    A = np.dot(np.transpose(p), q)
    V, s, W = np.linalg.svd(A)
    A_2 = np.dot(np.dot(V, np.diag(s)), W)
    d = np.sign(np.linalg.det(np.dot(np.transpose(W), np.transpose(V))))
    s_2 = np.ones(len(s))
    s_2[len(s)-1] = d

    # check whether rotation_mat is valid rotation matrix or not
    assert np.linalg.norm(np.identity(3) - np.dot(rotation_mat, np.transpose(rotation_mat))) < 1e-6

    return np.transpose(rotation_mat)

