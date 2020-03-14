#!/usr/bin/env python

import argparse
import os, sys, glob
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core.utils.config import Config
import core.utils.bvh_to_joint as btoj


def collect_path(dataset):
    npy_paths = []
    for (root, dirs, files) in os.walk(dataset):
        for npy_dir in dirs:
            if npy_dir.find('spline') > -1:
                continue 
            npy_paths.extend(glob.glob(os.path.join(root, npy_dir, '*.npy')))
        if not npy_paths:
            for npy_file in files:
                if not npy_file.endswith('.npy'):
                    continue 
                npy_paths.append(os.path.join(root, npy_file))

    return npy_paths 

def calcurate_minmax():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-i', default='~/M1/motiongan/data/CMU_jp/Locomotion_jp',
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--standard_bvh', required=True)
    args = parser.parse_args()


    # Prepare skelton
    skelton, non_end_bones, joints_to_index = btoj.get_standard_format(args.standard_bvh)
    _, non_zero_joint_to_index = btoj.cut_zero_length_bone(skelton, joints_to_index)

    head, ext = os.path.splitext(args.dataset)
    head, data_name = os.path.split(head)

    npy_paths = collect_path(args)

    datamin = np.inf
    datamax = - np.inf
    for npy_path in npy_paths:
        motion = np.load(npy_path)
        # Cut zero from motion
        _, motion = btoj.cut_zero_length_bone_frames(motion, skelton, joints_to_index)
        motion = motion[:,3:] 
        datamin = min(datamin, np.min(motion, axis=(0,1)))
        datamax = max(datamax, np.max(motion, axis=(0,1)))
        print(motion.shape, datamin, datamax)

    print(datamin)
    print(datamax)

    return datamin, datamax

def calcurate_minmax_eachjoint():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-i', default='~/M1/motiongan/data/CMU_jp/Locomotion_jp',
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    head, ext = os.path.splitext(args.dataset)
    head, data_name = os.path.split(head)

    npy_paths = collect_path(args)

    flag = 0
    data = []
    for npy_path in npy_paths:
        npy = np.load(npy_path)
        npy = np.reshape(npy, (npy.shape[0], npy.shape[1] * 3)).transpose(1,0)
        print(npy.shape)
        if flag == 0:
            data = npy
            flag = 1
        else:
            data = np.concatenate((data, npy), axis=1)

    print(data.shape)

    datamin = np.min(data, axis=1)
    datamax = np.max(data, axis=1)
    np.savez('minmax/minmax_{0}.npz'.format(data_name), min=datamin, max=datamax)

    print(datamin)
    print(datamax)

    return datamin, datamax

def calcurate_meanstd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True) 
    parser.add_argument('--out', required=True)
    parser.add_argument('--standard_bvh', required=True)
    args = parser.parse_args()

    npy_paths = collect_path(cfg.train.dataset)

    head, ext = os.path.splitext(cfg.train.dataset)
    head, data_name = os.path.split(head)


    # Prepare skelton
    skelton, non_end_bones, joints_to_index = btoj.get_standard_format(args.standard_bvh)
    _, non_zero_joint_to_index = btoj.cut_zero_length_bone(skelton, joints_to_index)

    data = None
    for npy_path in npy_paths:
        motion = np.load(npy_path)
        # Cut zero from motion
        _, motion = btoj.cut_zero_length_bone_frames(motion, skelton, joints_to_index)
        print(motion.shape)
        if data is None:
            data = motion
        else:
            data = np.concatenate((data, motion), axis=0)


    print('meanstd',data.shape)

    datamean = np.mean(data, axis=0)
    datastd = np.std(data, axis=0)
    np.savez(args.out, min=datamean, max=datastd)

    print(datamean.shape, datastd.shape)

    return datamean, datastd

def calcurate_style_meanstd():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    cfg = Config.from_file(args.config)
    npy_paths = collect_path(cfg.train.dataset)

    head, ext = os.path.splitext(cfg.train.dataset)
    head, data_name = os.path.split(head)


    # Prepare skelton
    skelton, non_end_bones, joints_to_index, permute_xyz_order = btoj.get_standard_format(cfg.standard_bvh)
    _, non_zero_joint_to_index = btoj.cut_zero_length_bone(skelton, joints_to_index)

    Lul_offset = np.array(skelton['LeftUpLeg']['offsets'])
    Rul_offset = np.array(skelton['RightUpLeg']['offsets'])
    Lul_index = non_zero_joint_to_index['LeftUpLeg']
    Rul_index = non_zero_joint_to_index['RightUpLeg']

    standard_vector = Lul_offset-Rul_offset
    standard_norm = np.sqrt((Lul_offset[0]-Rul_offset[0])**2+(Lul_offset[2]-Rul_offset[2])**2)

    # Initialize
    result = {style:{joint:0 for joint in non_zero_joint_to_index} for style in cfg.train.class_list}

    data = None
    for npy_path in npy_paths:
        motion = np.load(npy_path)
        # Cut zero from motion
        _, motion = btoj.cut_zero_length_bone_frames(motion, skelton, joints_to_index)

        # Convert trajectory to velocity
        motion = motion[1:]
        trajectory = motion[:,:3]
        velocity = trajectory[1:,:] - trajectory[:-1,:]
        motion = np.concatenate((velocity, motion[1:,3:]), axis=1)

        # Get orientation ('xz' only)
        motion_oriented = np.zeros_like(motion)
        leftupleg = motion[:,Lul_index*3:Lul_index*3+3]
        rightupleg = motion[:,Rul_index*3:Rul_index*3+3]
        vector = leftupleg-rightupleg
        norm = np.sqrt(vector[:,0]**2+vector[:,2]**2)

        cos = (vector[:,0]*standard_vector[0]+vector[:,2]*standard_vector[2]) / (norm*standard_norm)
        cos = np.clip(cos, -1, 1)
        sin = 1 - cos**2

        for t in range(motion.shape[0]):
            rotation_mat = np.array([[cos[t], 0., -sin[t]],
                                     [0.,  1.,   0.],
                                     [sin[t], 0.,  cos[t]]]) 
            motion_oriented[t,:] = np.dot(rotation_mat.T, motion[t,:].reshape(28,3).T).T.reshape(-1,)

        # Set class
        npy_name = os.path.splitext(os.path.split(npy_path)[1])[0]
        style = npy_name.split('_')[0]

        mean = np.mean(motion[1:], axis=0)
        std = np.std(motion[1:], axis=0)

        mean_oriented = np.mean(motion_oriented, axis=0)
        std_oriented = np.std(motion_oriented, axis=0)

        # Write
        for joint in non_zero_joint_to_index:
            ji = non_zero_joint_to_index[joint]
            result[style][joint] = {'mean': mean_oriented[ji*3:ji*3+3], 'std':std_oriented[ji*3:ji*3+3]}

    with open(args.out, 'wb') as f:
        pickle.dump(result, f)

    ##t-SNE
    mean_data, std_data, label_data = [], [], []
    for style in result.keys():
        data = result[style]
        mean_data.append([data[joint]['mean'] for joint in non_zero_joint_to_index])
        std_data.append([data[joint]['std'] for joint in non_zero_joint_to_index])
        label_data.append(cfg.train.class_list.index(style))
    mean_data = np.array(mean_data).reshape(len(mean_data),-1)
    std_data = np.array(std_data).reshape(len(mean_data),-1)
    label_data = np.array(label_data).reshape(len(mean_data),)


    plt.figure(figsize=(30,30), dpi=72)

    # joint mean
    mean_velocity = np.stack((mean_data[:,6],mean_data[:,7]), axis=1)

    plt.subplot(331)
    plt.scatter(mean_velocity[:,0], mean_velocity[:,1], s=25)
    plt.title('mean'+list(non_zero_joint_to_index.keys())[2])
    for i in range(mean_velocity.shape[0]):
        point = mean_velocity[i]
        label = cfg.train.class_list[label_data[i]]
        plt.text(point[0], point[1], label, fontsize=8)

    # joint std
    std_velocity = np.stack((std_data[:,6],std_data[:,7]), axis=1)

    plt.subplot(332)
    plt.scatter(std_velocity[:,0], std_velocity[:,1], s=25)
    plt.title('std_'+list(non_zero_joint_to_index.keys())[2])
    for i in range(std_velocity.shape[0]):
        point = std_velocity[i]
        label = cfg.train.class_list[label_data[i]]
        plt.text(point[0], point[1], label, fontsize=8)

    # PCA mean
    pca = PCA(n_components=2)
    mean_pca = pca.fit_transform(mean_data)

    plt.subplot(333)
    plt.scatter(mean_pca[:,0], mean_pca[:,1], s=25)
    plt.title('pca_mean')
    for i in range(mean_pca.shape[0]):
        point = mean_pca[i]
        label = cfg.train.class_list[label_data[i]]
        plt.text(point[0], point[1], label, fontsize=8)

    # joint mean
    mean_velocity = np.stack((mean_data[:,36],mean_data[:,37]), axis=1)

    plt.subplot(334)
    plt.scatter(mean_velocity[:,0], mean_velocity[:,1], s=25)
    plt.title('mean_'+list(non_zero_joint_to_index.keys())[12])
    for i in range(mean_velocity.shape[0]):
        point = mean_velocity[i]
        label = cfg.train.class_list[label_data[i]]
        plt.text(point[0], point[1], label, fontsize=8)

    # joint std
    std_velocity = np.stack((std_data[:,36],std_data[:,37]), axis=1)

    plt.subplot(335)
    plt.scatter(std_velocity[:,0], std_velocity[:,1], s=25)
    plt.title('std_'+list(non_zero_joint_to_index.keys())[12])
    for i in range(std_velocity.shape[0]):
        point = std_velocity[i]
        label = cfg.train.class_list[label_data[i]]
        plt.text(point[0], point[1], label, fontsize=8)

    # PCA mean
    pca = PCA(n_components=2)
    std_pca = pca.fit_transform(std_data)

    plt.subplot(336)
    plt.scatter(std_pca[:,0], std_pca[:,1], s=25)
    plt.title('pca_std')
    for i in range(std_pca.shape[0]):
        point = std_pca[i]
        label = cfg.train.class_list[label_data[i]]
        plt.text(point[0], point[1], label, fontsize=8)

    # joint mean
    mean_velocity = np.stack((mean_data[:,60],mean_data[:,61]), axis=1)

    plt.subplot(337)
    plt.scatter(mean_velocity[:,0], mean_velocity[:,1], s=25)
    plt.title('mean_'+list(non_zero_joint_to_index.keys())[20])
    for i in range(mean_velocity.shape[0]):
        point = mean_velocity[i]
        label = cfg.train.class_list[label_data[i]]
        plt.text(point[0], point[1], label, fontsize=8)

    # joint std
    std_velocity = np.stack((std_data[:,60],std_data[:,61]), axis=1)

    plt.subplot(338)
    plt.scatter(std_velocity[:,0], std_velocity[:,1], s=25)
    plt.title('std_'+list(non_zero_joint_to_index.keys())[20])
    for i in range(std_velocity.shape[0]):
        point = std_velocity[i]
        label = cfg.train.class_list[label_data[i]]
        plt.text(point[0], point[1], label, fontsize=8)


    # PCA mean and std
    pca = PCA(n_components=2)
    mean_std_pca = pca.fit_transform(np.concatenate((mean_data, std_data), axis=1))

    plt.subplot(339)
    plt.scatter(mean_std_pca[:,0], mean_std_pca[:,1], s=25)
    plt.title('pca_mean_std')
    for i in range(mean_std_pca.shape[0]):
        point = mean_std_pca[i]
        label = cfg.train.class_list[label_data[i]]
        plt.text(point[0], point[1], label, fontsize=8)

    plt.savefig(os.path.splitext(args.out)[0]+'_tSNE.png')


    # 3D PCA mean and std
    fig = plt.figure(figsize=(10,10), dpi=72)
    ax = Axes3D(fig)
    pca = PCA(n_components=3)
    mean_std_pca3d = pca.fit_transform(np.concatenate((mean_data, std_data), axis=1))

    ax.scatter3D(mean_std_pca3d[:,0], mean_std_pca3d[:,1], mean_std_pca3d[:,2], s=25)
    for i in range(mean_std_pca.shape[0]):
        point = mean_std_pca3d[i]
        label = cfg.train.class_list[label_data[i]]
        ax.text(point[0], point[1], point[2], label, fontsize=8)

    plt.savefig(os.path.splitext(args.out)[0]+'_PCA3D.png')


def calcurate_average_speed():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    cfg = Config.from_file(args.config)
    npy_paths = collect_path(cfg.train.dataset)

    head, ext = os.path.splitext(cfg.train.dataset)
    head, data_name = os.path.split(head)

    class_list = cfg.train.class_list

    v_traj_avg_dict = {label:0 for label in class_list}
    class_npy_paths = {label:[] for label in class_list}
    for npy_path in npy_paths:
        npy_name = os.path.splitext(os.path.split(npy_path)[1])[0]
        label = npy_name.split('_')[0]
        class_npy_paths[label].append(npy_path)

    for label in class_npy_paths.keys():
        class_v_traj = None
        for npy_path in class_npy_paths[label]:
            motion = np.load(npy_path)
            trajX = motion[:,0]
            trajZ = motion[:,2]
            v_trajX = trajX[:-1] - trajX[1:]
            v_trajZ = trajZ[:-1] - trajZ[1:]
            class_v_traj = np.sqrt(v_trajX**2 + v_trajZ**2) if class_v_traj is None else np.concatenate((class_v_traj, np.sqrt(v_trajX**2 + v_trajZ**2)), axis=0)
        if not class_npy_paths[label]:
            continue
        class_v_traj_avg = np.average(class_v_traj, axis=0)
        v_traj_avg_dict[label] = class_v_traj_avg
    print(v_traj_avg_dict)
 
    with open(args.out, 'wb') as f:
        pickle.dump(v_traj_avg_dict, f)
    
    return 


if __name__ == '__main__':
    #datamean, datastd = calcurate_meanstd()
    #calcurate_style_meanstd()
    #datamin, datamax = calcurate_minmax_eachjoint()
    calcurate_average_speed()
    #calcurate_minmax()
    
