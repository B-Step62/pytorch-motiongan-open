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



def apply_pca_content(gen, result_dir, target, n_components, cfg, device):
    gen.eval()


    ## Define class list
    class_list = cfg.train.class_list if hasattr(cfg.train, 'class_list') else [] 
    class_list = class_list
    num_class = len(class_list)
    use_label = num_class > 0

    ## Define content list
    content_list = cfg.train.content_list 
    content_list = content_list
    num_content = len(content_list)


    ## Inference Start
    if use_label:
        num_samples_per_class = 1

        ## Output of Latent Transform
        if target == 'w':
            data = {content_name:[] for content_name in content_list}
            label_data = {content_name:[] for content_name in content_list}
            label_list = {content_name:[] for content_name in content_list}

            for cls, class_name in enumerate(class_list):
                for cnt, content_name in enumerate(content_list):
                    class_label = torch.from_numpy(np.array([[cls]]).astype(np.int64)).to(device)
                    content_label = torch.from_numpy(np.array([[cnt]]).astype(np.int64)).to(device)

                    label_name = f'{class_list[cls]}_{content_list[cnt]}'
                    for i in range(num_samples_per_class):
                        z_i = gen.make_hidden(1, cfg.train.frame_nums//cfg.train.frame_step).to(device) if cfg.train.generator.use_z else None
                        # Generate w and embed from label
                        w_c_i = gen.latent_transform(z_i, class_label, content_label)
                        data[content_name].append(w_c_i[0,:,0,0].data.cpu().numpy())
                        label_data[content_name].append(tuple([cls*num_content+cnt, label_name]))
                    label_list[content_name].append(label_name)

            data['all'] = []
            label_data['all'] = []
            label_list['all'] = []
            for content_name in content_list:
                data['all'] += data[content_name] 
                label_data['all'] += label_data[content_name] 
                label_list['all'] += label_list[content_name] 
            
            data['all_sign'] = data['all'] 
            label_data['all_sign'] = label_data['all'] 
            label_list['all_sign'] = label_list['all'] 

            out_path = result_dir+f'/{target}_pca_{n_components}D.pdf'
            fit_and_plot(data, label_data, label_list, out_path, n_components,content_list, class_list, smallfont=True)

        ### Label embed output
        #elif target == 'embed':
        #    data = {'embed':[]}
        #    for c in range(num_class):
        #        label_c = torch.from_numpy(np.array([[c]]).astype(np.int64)).to(device)
        #        for i in range(num_samples_per_class):
        #            z_i = gen.make_hidden(1, cfg.train.frame_nums//cfg.train.frame_step).to(device) if cfg.train.generator.use_z else None
        #            # Generate w and embed from label
        #            if cfg.train.generator.norm == 'adain':
        #                label_embed_i, _ = gen.latent_transform(z_i, label_c, output_embed=True)
        #                data['embed'].append(label_embed_i[0,:,0,0].data.cpu().numpy())
        #            else:
        #                label_embed_i = gen.inference_embed(label_c)
        #                data['embed'].append(label_embed_i[0,0,:].data.cpu().numpy())
        #            class_data.append(c)
        #    out_path = result_dir+f'/{target}_pca_{n_components}D.pdf'
        #    fit_and_plot(data, class_data, class_list, out_path, n_components)

        ### Scale and Bias of each AdaIN layer
        #elif target.startswith('adain'):
        #    data = {'scale':[], 'bias':[]}
        #    for c in range(num_class):
        #        label_c = torch.from_numpy(np.array([[c]]).astype(np.int64)).to(device)
        #        for i in range(num_samples_per_class):
        #            z_i = gen.make_hidden(1, cfg.train.frame_nums//cfg.train.frame_step).to(device) if cfg.train.generator.use_z else None
        #            # Inference each parameters of each adain layer
        #            adain_params = gen.inference_adain(z_i, label_c, target)
        #            data['scale'].append(adain_params['scale'][0,:,0,0].data.cpu().numpy())
        #            data['bias'].append(adain_params['bias'][0,:,0,0].data.cpu().numpy())
        #            class_data.append(c)
        #    out_path = result_dir+f'/{target}_pca_{n_components}D.pdf'
        #    fit_and_plot(data, class_data, class_list,out_path, n_components)
        #        
                    

        





def fit_and_plot(data_list, label_data, label_list, out_path, n_components, content_list, class_list, smallfont=False, putText=True):


    putText=False

    if n_components == 2:
        graph_nums = len(list(data_list.keys())) 
        fig = plt.figure(figsize=(10*2, 10*(graph_nums+1)//2), dpi=216)
        for i, (name, data) in enumerate(data_list.items()):
            # FIt PCA model
            data = np.array(data)
            #pca = PCA(n_components=n_components, random_state=0)
            #pca.fit(data)
            #data_reduced = pca.transform(data)
            pca = PCA(n_components=3, random_state=0)
            pca.fit(data)
            data_reduced = pca.transform(data)[:,1:]
            print(f'Decomposing {name} : {data.shape} -> {data_reduced.shape}')
            print(f'Contribution ratio : {pca.explained_variance_ratio_}')
        
            num_samples_per_class = data.shape[0]//len(label_list[name])
    
            # Plot 
            plt.subplot((graph_nums+1)//2, 2, i+1)
            plt.title(name)


            for i in range(data_reduced.shape[0]):
                content = label_list[name][i//num_samples_per_class][-2:]
                if content == 'FW': marker = "o"
                elif content == 'FR': marker = "p"
                elif content == 'BW': marker = "v"
                elif content == 'BR': marker = "^"
                elif content == 'SW': marker = "+"
                elif content == 'SR': marker = "x"
                plt.scatter(data_reduced[i:i+1,0], data_reduced[i:i+1,1], s=100, marker=marker, c=[cm.hsv(cl/len(label_list['all'])) for cl,name in label_data[name][i:i+1]])

          
            for c, label in enumerate(label_list[name]):
                center = np.mean(data_reduced[c*num_samples_per_class:(c+1)*num_samples_per_class,:], axis=0)
                if putText:
                    plt.text(center[0], center[1], label_list[name][c], fontsize=8 if smallfont else 16, alpha=0.8)

            # Draw path between same class diferrent content
            if name == 'all_sign':
                centers_of_class = {cls : [] for cls in class_list}
                for c, label in enumerate(label_list[name]):
                    center = np.mean(data_reduced[c*num_samples_per_class:(c+1)*num_samples_per_class,:], axis=0)
                    cls = label.split('_')[0]
                    centers_of_class[cls].append(center)
 
                for c, (cls, centers) in enumerate(list(centers_of_class.items())):
                    centers.append(centers[0])
                    centers = np.array(centers)
                    plt.plot(centers[:,0], centers[:,1], c=cm.hsv(c/len(class_list)), lw=1)

    elif n_components == 3:
        graph_nums = len(list(data_list.keys())) 
        fig = plt.figure(figsize=(10*graph_nums, 10), dpi=216)
        for i, (name, data) in enumerate(data_list.items()):
            # FIt PCA model
            data = np.array(data)
            pca = PCA(n_components=n_components, random_state=0)
            pca.fit(data)
            data_reduced = pca.transform(data)
            print(f'Decomposing {name} : {data.shape} -> {data_reduced.shape}')
            print(f'Contribution ratio : {pca.explained_variance_ratio_}')
        
            num_samples_per_class = data.shape[0]//len(label_list[name])
    
            # Plot 
            plt.subplot(1, graph_nums, i+1)
            plt.title(name)

            ax = Axes3D(fig)
            ax.view_init(elev=20, azim=-30)
            ax.scatter3D(data_reduced[:,0], data_reduced[:,2], data_reduced[:,1], s=50, c=[cm.hsv(cl/len(label_list)) for cl, name in label_data[name]])
            if label_list:
                for c in range(len(label_list)):
                    center = np.mean(data_reduced[c*num_samples_per_class:(c+1)*num_samples_per_class,:], axis=0)
                    ax.text(center[0], center[2], center[1], label_list[c], fontsize=12, alpha=0.5)


    plt.savefig(out_path)
    plt.close()
