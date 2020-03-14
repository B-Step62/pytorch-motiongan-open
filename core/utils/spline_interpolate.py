import sys
import numpy as np
from scipy import interpolate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SplineInterpolate:
    def __init__(self, dt=0.01, kind='cubic', sampling_step=1):
        self.dt = dt
        self.kind = kind
        self.sampling_step = sampling_step

    def interpolate_spline(self, trajectory, spline_step):
        t = np.linspace(0, trajectory.shape[0]-1, trajectory.shape[0]//spline_step)
        f = interpolate.interp1d(t, trajectory[::spline_step,:], kind=self.kind, axis=0)
        return f

    def get_spline_length(self, trajectory, f):
        # calculate whole spline length
        t = 0
        length = 0
        fN = (trajectory.shape[0]-1) / self.dt - 1
        lenmap = dict()
        while t < fN:
            lenmap[t] = length
            ds = f((t+1)*self.dt) - f(t*self.dt)
            if len(ds) == 3:
                length += np.sqrt(ds[0]**2 + ds[1]**2 + ds[2]**2)
            elif len(ds) == 2:
                length += np.sqrt(ds[0]**2 + ds[1]**2)
            t += 1 
        return lenmap
 
    def sampling(self, trajectory, f, lenmap, startT=0, endT=None, step=1):
        # trajectory : root positions extracted from motion
        # f : spline function obtained by interp1d
        # lenmap : hashmap between t(dt) to length of spline
        #           e.g.  t=0.00 : 0
        #                 t=0.01 : 0.1
        #                 t=0.02 : 0.2
        # startT : start frame index
        # endT : end frame index
        # step : frame step (same as frame_step defined in config)
        if endT is None:
            endT = trajectory.shape[0]
        # calcurate equal length interval between startT and endT
        interval = (lenmap[int((endT-1)/self.dt)] - lenmap[int(startT/self.dt)]) / ((endT - startT) // step - 1) 
        # initialize t as startT and fN as endT 
        t = int(startT/self.dt) 
        fN = (endT-1)/self.dt-1
        # initialize 'current' as lenmap[startT/dt]
        current = lenmap[t] 
        # 'samples' is sampling trajectory to return
        samples = [trajectory[startT]]
        while len(samples) < (endT-startT)//step-1 and t < fN:
            let = lenmap[t+1]
            # if current length at frame t over the interval point, add its coordinate to samples
            if current + interval < let:
                samples.append(f((t+1)*self.dt))
                current += interval
            t += 1 
        samples.append(trajectory[endT-1])
        return np.array(samples)

    def show(self, trajectory, samples, f):
        avex = np.average(trajectory[:,0], axis=0)
        avey = np.average(trajectory[:,2], axis=0)
        avez = np.average(trajectory[:,1], axis=0)
        rangemax = 5
        frame_length = trajectory.shape[0]


        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=15, azim=15)

        ax.set_xlim(avex-rangemax//2, avex+rangemax//2)
        ax.set_ylim(avey-rangemax//2, avey+rangemax//2)
        ax.set_zlim(avez-rangemax//2, avez+rangemax//2)

   
        spline = np.array([f(t/50) for t in range((trajectory.shape[0]-1)*50)])

        ax.scatter3D(spline[:,0], spline[:,2], spline[:,1]-1.0, color='green', marker='.')
        ax.scatter3D(trajectory[:,0], trajectory[:,2], trajectory[:,1], color='red', marker='.')
        ax.scatter3D(samples[:,0], samples[:,2], samples[:,1]+1.0, color='blue', marker='.')

        plt.savefig(f'sampletest.png')

