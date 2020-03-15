import os, sys, glob
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from core import models
from core.utils.config import Config


### heatmap of distance between a pair of centroids of each class
def draw_heatmap(cfg, gen, result_dir, device, num_samples_per_class=50):


    ## Define class list
    class_list = cfg.train.dataset.class_list
    num_class = len(class_list)


    ## Inference Start
    centroids = {cls_name:0 for cls_name in class_list}
    
    # Target: Output of Latent Transform
    for c in range(num_class):
        label = torch.from_numpy(np.array([[c]]).astype(np.int64)).to(device)
        center = 0
        for _ in range(num_samples_per_class):
            z = gen.make_hidden(1, cfg.train.dataset.frame_nums//cfg.train.dataset.frame_step).to(device) if cfg.models.generator.use_z else None
            # Generate w and embed from label
            w  = gen.latent_transform(z, label)
            center += w[0,:,0,0].data.cpu().numpy()
        center /= num_samples_per_class
        centroids[class_list[c]] = center

    distance = {}
    for cls_name in class_list:
        distance[cls_name] = [np.average((centroids[cls_name]-centroids[target_cls_name])**2) for target_cls_name in class_list]
    
        

    # Convert data to pandas dataframe
    distance_df = pd.DataFrame(data=distance, index=class_list)
    print(distance_df)
         

    # Calc top 5
    top5 = {}
    with open(os.path.join(result_dir, f'centroid_top5.txt'), 'w') as f:
        for cls_name in class_list:
            top = np.argsort(distance[cls_name]).tolist()
            top5[cls_name] = [class_list[cls] for cls in top[1:6]]
            f.write(f'{cls_name} : '+','.join(top5[cls_name])+'\r\n')

    # Draw heatmap
    plt.figure(figsize=(30,30), dpi=72)
    seaborn.heatmap(distance_df, square=True, annot=True, cmap='Blues')
    plt.savefig(os.path.join(result_dir, f'centroid_heatmap.png'))

    distance_df.to_csv(os.path.join(result_dir, f'centroid_distance.csv'))
