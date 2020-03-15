import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from sklearn.decomposition import PCA
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn



def apply_pca(cfg, gen, result_path, target, components, device, num_samples_per_class=10):

    ## Define class list
    class_list = cfg.train.dataset.class_list 
    num_class = len(class_list)


    ## Inference Start
    label_data = []

    # Target: Output of Latent Transform
    if target == 'w':
        target_data = {'w':[]}
        for c in range(num_class):
            label = torch.from_numpy(np.array([[c]]).astype(np.int64)).to(device)
            for _ in range(num_samples_per_class):
                z = gen.make_hidden(1, cfg.train.dataset.frame_nums//cfg.train.dataset.frame_step).to(device) if cfg.models.generator.use_z else None
                # Generate w and embed from label
                w = gen.latent_transform(z, label)
                target_data['w'].append(w[0,:,0,0].data.cpu().numpy())
                label_data.append(c)


    # Target: Scale and Bias of each AdaIN layer
    elif target.startswith('adain'):
        target_data = {'scale':[], 'bias':[]}
        for c in range(num_class):
            label = torch.from_numpy(np.array([[c]]).astype(np.int64)).to(device)
            for _ in range(num_samples_per_class):
                z = gen.make_hidden(1, cfg.train.dataset.frame_nums//cfg.train.dataset.frame_step).to(device) if cfg.models.generator.use_z else None
                # Inference each parameters of each adain layer
                adain_params = gen.inference_adain(z, label, target)
                target_data['scale'].append(adain_params['scale'][0,:,0,0].data.cpu().numpy())
                target_data['bias'].append(adain_params['bias'][0,:,0,0].data.cpu().numpy())
                label_data.append(c)
            

    ## Fit PCA and plot            
    gcols, grows = len(list(target_data.keys())), len(components)
    fig = plt.figure(figsize=(10*gcols, 10*grows), dpi=216)

    # Get maximum component 
    max_component = max([max(p) for p in components])

    for i, (target, data) in enumerate(target_data.items()):
        # FIt PCA model
        data = np.array(data)
        pca = PCA(n_components=max_component, random_state=0)
        pca.fit(data)
        data_reduced = pca.transform(data)
        print(f'Decomposing \033[1m{target}\033[0m : {data.shape} -> {data_reduced.shape}')
        print(f'Contribution ratio : {pca.explained_variance_ratio_}')

        # Plot each pair of components 
        for j, (x, y) in enumerate(components):
            plt.subplot(grows, gcols, gcols*j+i+1)
            plt.title(target)
            plt.xlabel(f'{x}th principal')
            plt.ylabel(f'{y}th principal')

            # Scatter points
            plt.scatter(data_reduced[:,x-1], data_reduced[:,y-1], s=30, c=[cm.hsv(l/len(class_list)) for l in label_data])
            # Plot class name at center of each cluster
            for c in range(len(class_list)):
                center = np.mean(data_reduced[c*num_samples_per_class:(c+1)*num_samples_per_class,:], axis=0)
                plt.text(center[x-1], center[y-1], class_list[c], fontsize=8, alpha=0.7)

    plt.savefig(result_path)
    plt.close()
