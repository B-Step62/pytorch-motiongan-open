import math

import numpy as np
from scipy import interpolate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn.functional as F

from core.utils.bvh_to_joint import get_standard_format, collect_bones

# Permute motion rows
def permute_motion(motion, order):
    motion_origin = motion.clone()
    for r1, r2 in enumerate(order):
        motion[:,:,r2*3:(r2+1)*3] = motion_origin[:,:,r1*3:(r1+1)*3]
    return motion


# Rotate motion around xy axis
def rotate(motion, theta):
    if motion.ndim == 2:
        motion = motion.reshape((motion.shape[0], -1, 3))
    # motion shape time * pose * xyz
    rotation_mat = np.matrix([[np.cos(theta), 0., -np.sin(theta)],
                              [0.,            1.,             0.],
                              [np.sin(theta), 0.,  np.cos(theta)]])                         

    motion_rot = np.zeros(motion.shape)
    for t in range(motion.shape[0]):
        motion_rot[t,:,:] = np.dot(rotation_mat, motion[t,:,:].T).T
    motion_rot = motion_rot.reshape((motion_rot.shape[0], -1))        
    return motion_rot

# Randomly rotate motion around xy axis
def random_rotate(motion):
    theta = np.random.rand() * 2. * math.pi
    return rotate(motion, theta)

# Cut a fixed number of frames from motion except center hint frame
def cutout(motion, index, cutrange=33):
    lack_motion = motion.clone()
    if cutrange < 2:
        return lack_motion
    cutstart = max(0, index-cutrange//2)
    cutend = min(motion.shape[2], index+cutrange//2+1)
    if index > 0:
        lack_motion[:,:,cutstart:index,:] = torch.zeros((motion.shape[0], motion.shape[1], index-cutstart, motion.shape[3]))
    if index < motion.shape[2] - 1:
        lack_motion[:,:,index+1:cutend,:] = torch.zeros((motion.shape[0], motion.shape[1], cutend-index-1, motion.shape[3]))
    return lack_motion


# Calcurate velocity from trajectory
def get_velocity(trajectory):
    v_trajectory = trajectory[:,:,1:,:] - trajectory[:,:,:-1,:]
    v_trajectory = F.pad(v_trajectory, (0,0,1,0), mode='reflect')
    return v_trajectory

# Reconstruct trajectory from verocity
def reconstruct_v_trajectory(v_trajectory, start_position):
    fake_trajectory = start_position 
    for t in range(1, v_trajectory.shape[2]):
        fake_trajectory = torch.cat((fake_trajectory, v_trajectory[:,:,t:t+1,:] + fake_trajectory[:,:,t-1:t,:]), dim=2)
    return fake_trajectory

# Smoothing using convolution
def smoothing(motion, window_size):
    motion_smooth = np.zeros((motion.shape[0]-window_size+1, 3))
    for j in range(motion.shape[1]):
        window = np.ones(window_size) / window_size
        motion_smooth[:,j] = np.convolve(motion[:,j], window, mode='valid')
    return motion_smooth

# Get orientation (direction of body) as normal vector of front body plane
def get_front_orientation(motion):
    # root is center, so these two coordinate is also represent vectors from root 
    left_shoulder = motion[16*3:17*3,:]
    right_shoulder = motion[22*3:23*3,:]

    # then, cross product is normal vector of front body plane
    orientation = np.array([left_shoulder[1,:]*right_shoulder[2,:] - left_shoulder[2,:]*left_shoulder[1,:],
              left_shoulder[2,:]*right_shoulder[0,:] - left_shoulder[0,:]*left_shoulder[2,:],
              left_shoulder[0,:]*right_shoulder[1,:] - left_shoulder[1,:]*left_shoulder[0,:]])

    return orientation

# make direction label sequense according to orientation
def labeling_orientation(orientation, label_num):
    sinxy = orientation[2,:] / orientation[1,:]
    cosxy = orientation[2,:] / orientation[1,:]
    theta = np.arccos(cosxy)
    if sinxy < 0:
        theta += math.pi
    
    # TODO
    return label 
     


## Spline generation
# Apply spline interpolation to trajectory 
def interpolate_spline(trajectory, control_point_interval, kind='cubic'):
    t = np.array([control_point_interval*i for i in range(int(np.ceil(trajectory.shape[0]/control_point_interval)))]+[trajectory.shape[0]-1])
    try:
        f = interpolate.interp1d(t, np.concatenate((trajectory[::control_point_interval,:], trajectory[-2:-1,:]), axis=0), kind=kind, axis=0)
    except ValueError:
        try:
            f = interpolate.interp1d(t, np.concatenate((trajectory[::control_point_interval,:], trajectory[-2:-1,:]), axis=0), kind='quadratic', axis=0)
        except ValueError:
            #print('One file ignored with the error below.')
            #print('ValueError: The number of derivatives at boundaries does not match: expected 2, got 0+0')
            f = None
    return f

# Get spline curve length and get mapping between variable t and length
def get_spline_length(trajectory, f, dt):
    if isinstance(f, np.ndarray):
        f_array = f.copy()
        f = lambda t: f_array[int(t/dt)] 
    t = 0
    length = 0
    fN = (trajectory.shape[0]-1) / dt
    lenmap = dict()
    while t < fN:
        lenmap[t] = length
        ds = f((t+1)*dt) - f(t*dt)
        if len(ds) == 3:
            length += np.sqrt(ds[0]**2 + ds[1]**2 + ds[2]**2)
        elif len(ds) == 2:
            length += np.sqrt(ds[0]**2 + ds[1]**2)
        t += 1
    lenmap[t] = length
    return lenmap

# Sampling points with equal interval
def sampling(trajectory, f, lenmap, dt, startT=0, endT=None, step=1, with_noise=False):
    # trajectory : root positions extracted from motion
    # f : numpy array or spline function obtained by interp1d
    # lenmap : hashmap between t(dt) to length of spline
    #           e.g.  t=0.00 : 0
    #                 t=0.01 : 0.1
    #                 t=0.02 : 0.2
    # startT : start frame index
    # endT : end frame index
    # step : frame step (same as frame_step defined in config)
    if endT is None:
        endT = trajectory.shape[0]
    if isinstance(f, np.ndarray):
        f_array = f.copy()
        f = lambda t: f_array[int(t/dt)] 
    # calcurate equal length interval between startT and endT
    interval = (lenmap[int((endT-1)/dt)] - lenmap[int(startT/dt)]) / ((endT - startT) // step - 1)
    # initialize t as startT and fN as endT 
    t = int(startT/dt)
    fN = len(lenmap)-1
    # initialize 'current' as lenmap[startT/dt]
    current = lenmap[t]
    # 'samples' is sampling trajectory to return
    samples = [trajectory[startT]]
    while len(samples) < (endT-startT)//step and t < fN:
        let = lenmap[t+1]
        # if current length at frame t over the interval point, add its coordinate to samples
        if current + interval < let:
            if with_noise:
                # timing noise
                timing_noise = (step/dt)*0.02*np.random.randn()
                point = f(min(fN, (max(0, timing_noise+t+1)))*dt)
                # spacial noise
                spacial_noise = interval*0.02*np.random.randn(2)
                point[0]+=spacial_noise[0]
                point[2]+=spacial_noise[1]
                samples.append(point)
            else:
                samples.append(f((t+1)*dt))
            current += interval
        t += 1
    while len(samples) < (endT-startT)//step:
        samples.append(trajectory[endT-1])
    return np.array(samples)

# Sampling points with equal time interval
def sampling_by_time(trajectory, f,startT=0, endT=None, step=1):
    # trajectory : root positions extracted from motion
    # f : numpy array or spline function obtained by interp1d
    # startT : start frame index
    # endT : end frame index
    # step : frame step (same as frame_step defined in config)
    if endT is None:
        endT = trajectory.shape[0]
    if isinstance(f, np.ndarray):
        f_array = f.copy()
        f = lambda t: f_array[int(t/dt)] 
    # 'samples' is sampling trajectory to return
    samples = []
    for t in range(startT, endT, step):
        samples.append(f(t))
    while len(samples) < (endT-startT)//step:
        samples.append(trajectory[endT-1])
    return np.array(samples)

# Sampling points with equal interval
def sampling_fixed(trajectory, f, lenmap, dt, interval, startT, frame_nums, step=1, with_noise=False):
    # trajectory : root positions extracted from motion
    # f : numpy array or spline function obtained by interp1d
    # lenmap : hashmap between t(dt) to length of spline
    #           e.g.  t=0.00 : 0
    #                 t=0.01 : 0.1
    #                 t=0.02 : 0.2
    # startT : start frame index
    # step : frame step (same as frame_step defined in config)
    endT = trajectory.shape[0]-1
    if isinstance(f, np.ndarray):
        f_array = f.copy()
        f = lambda t: f_array[int(t/dt)] 
    # initialize t as startT and fN as endT 
    t = int(startT/dt)
    fN = len(lenmap)-1
    # initialize 'current' as lenmap[startT/dt]
    current = lenmap[t]
    # 'samples' is sampling trajectory to return
    samples = [trajectory[startT]]
    while t+1 < fN and len(samples)<frame_nums:
        let = lenmap[t+1]
        # if current length at frame t over the interval point, add its coordinate to samples
        if current + interval < let:
            if with_noise:
                noise = (step/dt)*0.2*np.random.randn()
                samples.append(f(min(fN, (max(0, noise+t+1)))*dt))
            else:
                samples.append(f((t+1)*dt))
            current += interval
        t += 1
    end_frame = t*dt
    while len(samples) < frame_nums:
        samples.append(trajectory[-1])
    return np.array(samples), end_frame

# Low-pass filter
def low_pass_filtering(f, f_domain, f_cut=10.):
    # f : function reperesented as np-array
    dt = 1e-5
    t = np.arange(0, f_domain*dt, dt)
    freq = np.linspace(0, 1.0/dt, f_domain)

    # paramter 
    f_sample = 1.0/dt 
    f_upper = f_sample - f_cut 

    # calcurate with each axis
    g = np.zeros(f.shape)
    for axis in range(f.shape[1]):
        f_a = f[:,axis]

        # fft
        F_a = np.fft.fft(f_a)
        G_a = F_a.copy()

# Low-pass filter
def low_pass_filtering(f, f_domain, f_cut=10.):
    # f : function reperesented as np-array
    dt = 1e-5
    t = np.arange(0, f_domain*dt, dt)
    freq = np.linspace(0, 1.0/dt, f_domain)

    # paramter 
    f_sample = 1.0/dt 
    f_upper = f_sample - f_cut 

    # calcurate with each axis
    g = np.zeros(f.shape)
    for axis in range(f.shape[1]):
        f_a = f[:,axis]

        # fft
        F_a = np.fft.fft(f_a)
        G_a = F_a.copy()

        # low pass
        G_a[((freq>f_cut)&(freq<f_upper))] = 0 + 0j

        # inverse fft and get real value
        g_a = np.fft.ifft(G_a)
        g_a = g_a.real
        g[:,axis] = g_a

        #plt.subplot(221)
        #plt.plot(t, f_a)
        #plt.subplot(222)
        #plt.plot(freq, F_a)
        #plt.subplot(223)
        #plt.plot(t, g_a)
        #plt.subplot(224)
        #plt.plot(freq, G_a)
        #plt.savefig(f'testlp/axis{axis}.png')
        #plt.clf()
    return g 


# Apply low pass filter to spline
def apply_low_pass_to_spline(spline_f, f_domain, dt):
    # Calcurate velocity of spline
    v_spline_f = lambda x: spline_f(x+dt-1e-10)-spline_f(x) 
    
    # Apply low pass filter
    t_axis = np.linspace(0, f_domain*dt, f_domain+1)
    v_spline_f_array = v_spline_f(t_axis[:-1])
    v_spline_f_lp_array = low_pass_filtering(v_spline_f_array, f_domain)

    # Reconstruct trajectory from velocity 
    spline_f_lp_array = spline_f(t_axis)
    for t in range(v_spline_f_lp_array.shape[0]):
        spline_f_lp_array[t+1] = spline_f_lp_array[t] + v_spline_f_lp_array[t]

    return spline_f_lp_array

# Get norm(squared) between two index
def get_bone_norm(motion, bone, permute_order=None):
    ## Root position is not inculded in array (because it's always zeros).
    b0 = permute_order[bone[0]] if permute_order is not None and bone[0]>0 else bone[0]
    b1 = permute_order[bone[1]] if permute_order is not None and bone[1]>0 else bone[1]
    j1 = (b0-1)*3
    j2 = (b1-1)*3
    if j1 < 0: # Root
        x = motion[:,:,:,j2:j2+1]**2
        y = motion[:,:,:,j2+1:j2+2]**2
        z = motion[:,:,:,j2+2:j2+3]**2
    elif j2 < 0: # Root
        x = motion[:,:,:,j1:j1+1]**2
        y = motion[:,:,:,j1+1:j1+2]**2
        z = motion[:,:,:,j1+2:j1+3]**2
    else:
        x = (motion[:,:,:,j1:j1+1] - motion[:,:,:,j2:j2+1]) ** 2 
        y = (motion[:,:,:,j1+1:j1+2] - motion[:,:,:,j2+1:j2+2]) ** 2 
        z = (motion[:,:,:,j1+2:j1+3] - motion[:,:,:,j2+2:j2+3]) ** 2 
    return torch.sqrt(x+y+z)

def get_bones_norm(frames, bones, permute_order=None):
    bones_norm = None
    for bone in bones:
        bones_norm = get_bone_norm(frames, bone, permute_order) if bones_norm is None else torch.cat((bones_norm, get_bone_norm(frames, bone, permute_order)), dim=3)
    return bones_norm

## Get target norm of each bone in skeleton to calcurate bone loss (constraint loss)
def get_target_bones_norm(standart_bvh, train_dataset):
    bones = collect_bones(standart_bvh)
    standard_frame = torch.from_numpy(train_dataset[0][0][None,None,:,:]).type(torch.FloatTensor)
    target_bones_norm = get_bones_norm(standard_frame[:,:,:1,3:], bones)
    return target_bones_norm, bones



'''
1. At least one foot must not move because it is pivot.
2. Decide pivot foot as joint which has less movement.
3. Skating length can be obtained as movement of thus chosen joint.
'''
def calcurate_footskate(trajectory, motion_rightleg, motion_leftleg, threshold=0.5):
    total_footskate = torch.zeros(1)
    abs_rightleg = trajectory + motion_rightleg
    abs_leftleg = trajectory + motion_leftleg
    for t in range(trajectory.shape[0]-1):
        vec_right = abs_rightleg[t+1,:] - abs_rightleg[t,:]
        vec_left = abs_leftleg[t+1,:] - abs_leftleg[t,:]
        v_right = torch.sqrt(vec_right[0]**2+vec_right[2]**2) 
        v_left = torch.sqrt(vec_left[0]**2+vec_left[2]**2) 
        footskate = min(v_right, v_left)
        if footskate > threshold:
            total_footskate += footskate
    return total_footskate.item()

'''
Find nearest point in target_curve, from each trajecotry point
interval : calcurate point interval
nn_range : how many points of target_curve are included in NN candidates
'''
def calcurate_trajectory_error(trajectory, target_curve, interval, nn_range):
    total_dist = 0
    assert len(trajectory.shape) == 2 and len(target_curve.shape) == 2
    for t in range(0, trajectory.shape[0], interval):
        candidate = target_curve[max(0,t-nn_range//2):t+nn_range//2,:]
        nn_dist = torch.min((trajectory[t,0]-candidate[:,0])**2+(trajectory[t,2]-candidate[:,2])**2)
        nn_dist = torch.sqrt(nn_dist).item()
        total_dist += nn_dist
    return total_dist / (trajectory.shape[0]//interval)


def convert_event_to_spline(trajectory_raw, control_point_interval, dt=0.01):
    trajectory_3d = np.zeros((len(trajectory_raw), 3), dtype=np.float32)
    trajectory_x = np.array(trajectory_raw)[:,0]
    trajectory_y = np.array(trajectory_raw)[:,1]
    max_x = np.max(trajectory_x)
    max_y = np.max(trajectory_y)
    trajectory_3d[:,0] = trajectory_x / 10. - 2.5
    trajectory_3d[:,2] = trajectory_y / 10. - 2.5
    #print(trajectory_3d)

    spline_f = interpolate_spline(trajectory_3d, control_point_interval=control_point_interval)
    # apply low pass filter
    #f_domain = (trajectory_3d.shape[0]-1)/dt
    #spline_f = motion_utils.apply_low_pass_to_spline(spline_f, f_domain, dt)

    # Get length map for sampling
    spline_length_map = get_spline_length(trajectory_3d, spline_f, dt)
    samples = sampling(trajectory_3d, spline_f, spline_length_map, dt, step=1)

    #show(trajectory_3d, samples, spline_f, dt)

    return samples
