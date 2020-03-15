import os
import argparse
import random

import math
import pickle
import numpy as np
from scipy import interpolate
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.backends.cudnn as cudnn

from core import models
from core.utils.config import Config
from core.utils.XYZnormalize import Normalize
import core.utils.motion_utils as motion_utils

EVENT_PER_FRAMES = 2
PALETTE_SIZE = 3000
WINDOW_SIZE = 256

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

MARGIN = 100
WINDOW_MOVE_SPEED = 2

THICKNESS = 5
RADIUS = 3

is_drawing = False
is_erasing = False
end_flag = False

def draw_callback(event, x, y, flag, param):
    global is_drawing, is_erasing, end_flag

    px = min(max(THICKNESS+MARGIN, x + offset[0]), PALETTE_SIZE-THICKNESS+MARGIN)
    py = min(max(THICKNESS+MARGIN, y + offset[1]), PALETTE_SIZE-THICKNESS+MARGIN)
    
    # Draw while left button pushed down
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True

        palette[py-THICKNESS:py+THICKNESS, px-THICKNESS:px+THICKNESS] = 0
        trajectory_raw.append([px,py])

    elif is_drawing and event == cv2.EVENT_MOUSEMOVE:
        palette[py-THICKNESS:py+THICKNESS, px-THICKNESS:px+THICKNESS] = 0
        trajectory_raw.append([px,py])

    elif is_drawing and event == cv2.EVENT_LBUTTONUP:
        palette[py-THICKNESS:py+THICKNESS, px-THICKNESS:px+THICKNESS] = 0
        trajectory_raw.append([px,py])
        
        is_drawing = False
        end_flag = True

    # Erase while right button pushed down
    elif event == cv2.EVENT_RBUTTONDOWN:
        is_erasing = True
        cv2.circle(palette, (offset[0]+x, offset[1]+y), RADIUS, WHITE, THICKNESS, lineType=cv2.LINE_AA)

    elif event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(palette, (offset[0]+x, offset[1]+y), RADIUS, WHITE, THICKNESS, lineType=cv2.LINE_AA)

    # Move Window
    if 0<=offset[1]<=PALETTE_SIZE+MARGIN*2-WINDOW_SIZE:
        if y+THICKNESS > WINDOW_SIZE:
            offset[1] = min(int(offset[1]+WINDOW_MOVE_SPEED), PALETTE_SIZE+MARGIN*2-WINDOW_SIZE)
        elif y-THICKNESS < 0:
            offset[1] = max(int(offset[1]-WINDOW_MOVE_SPEED), 0)
    if 0<=offset[0]<=PALETTE_SIZE+MARGIN*2-WINDOW_SIZE:
        if x+THICKNESS > WINDOW_SIZE:
            offset[0] = min(int(offset[0]+WINDOW_MOVE_SPEED), PALETTE_SIZE+MARGIN*2-WINDOW_SIZE)
        elif x-THICKNESS < 0:
            offset[0] = max(int(offset[0]-WINDOW_MOVE_SPEED), 0)



def convert_to_spline(trajectory_raw, dt=0.01):
    trajectory_3d = np.zeros((len(trajectory_raw), 3), dtype=np.float32)
    trajectory_x = np.array(trajectory_raw)[:,0]
    trajectory_y = np.array(trajectory_raw)[:,1]
    max_x = np.max(trajectory_x)
    max_y = np.max(trajectory_y)
    trajectory_3d[:,0] = trajectory_x / 10. - 2.5
    trajectory_3d[:,2] = trajectory_y / 10. - 2.5

    spline_f = motion_utils.interpolate_spline(trajectory_3d)
    # apply low pass filter
    f_domain = (trajectory_3d.shape[0]-1)/dt
    spline_f = motion_utils.apply_low_pass_to_spline(spline_f, f_domain, dt)

    # Get length map for sampling
    spline_length_map = motion_utils.get_spline_length(trajectory_3d, spline_f, dt)
    samples = motion_utils.sampling(trajectory_3d, spline_f, spline_length_map, dt, step=EVENT_PER_FRAMES)

    #show(trajectory_3d, samples, spline_f, dt)

    return samples


def main():
    global palette, trajectory_raw, offset
    

    ### Get User Input #######

    # Initial pic
    palette = np.zeros([PALETTE_SIZE+MARGIN*2, PALETTE_SIZE+MARGIN*2], dtype=np.uint8) + 255
    # Edge of palette
    palette[:MARGIN,:] = 100
    palette[-MARGIN:,:] = 100
    palette[:,:MARGIN] = 100
    palette[:,-MARGIN:] = 100

    trajectory_raw = []

    # Create window and set callback
    cv2.namedWindow('overview', cv2.WINDOW_NORMAL)
    cv2.moveWindow('overview', 200,700)
    cv2.namedWindow('palette', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('palette', draw_callback)

    # Coordinate of left-up corner o f window
    offset = [PALETTE_SIZE//2-WINDOW_SIZE//2+MARGIN, PALETTE_SIZE//2-WINDOW_SIZE//2+MARGIN]

    prex = None
    prey = None

    # Get input
    while True and not end_flag:
        cv2.imshow('palette', palette[offset[1]:offset[1]+WINDOW_SIZE,offset[0]:offset[0]+WINDOW_SIZE])
        cv2.imshow('overview', cv2.resize(palette, (256, 256))) 

        key = cv2.waitKey(5)

       ## Key bindings

        # Exit
        if key == 27 or key == ord('q'):
            break

        # Reset
        if key == ord('r'):
            print('Reset window...\n')
            palette = np.zeros([PALETTE_SIZE+MARGIN*2, PALETTE_SIZE+MARGIN*2], dtype=np.uint8) + 255
            trajectory_raw = []


        # Too long trajectory
        if len(trajectory_raw) > 3000:
            break

    palette = palette[MARGIN:-MARGIN+1,MARGIN:-MARGIN+1]
           
    cv2.destroyAllWindows()

    # Get spline from input
    spline = convert_to_spline(trajectory_raw)

    with open('tmp_drawing.pkl', mode='wb') as f:
        data = {'spline':spline, 'palette':palette}
        pickle.dump(data, f)
    
    print('tmp_drawing.pkl')
    cv2.imwrite('drawing.png', palette) 



if __name__ == '__main__':
    main()

