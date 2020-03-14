import os, glob

import numpy as np

#['GlobalRoothipPos  Root hips  LHipJoint LeftUpLeg LeftLeg LeftFoot LeftToeBase EndSite RHipJoint']

class Normalize():
    def parse_initial_pos(self,bvh_path):
        offsets = []
        with open(bvh_path, 'r') as f:
            line = f.readline()
            while line:
                if 'OFFSET' in line:
                    pos = line.strip().split(' ')[1:]
                    pos = list(map(float, pos))
                    offsets.append(pos)
                line = f.readline()
        offsets = np.array(offsets)
        return offsets


    def readtxt(self,text_path):
        with open(text_path, 'r') as f:
            positions = []
            line = f.readline()
            while line:
                pos = line.strip().split('  ')
                pos = [tuple(map(float, p.split(' '))) for p in pos]
                #positions_unique = [np.array(list(positions[0]))]
                #pos[0] = np.array(list(pos[0]))
                #for i in range(1,len(pos)):
                #    pos[i] = np.array(list(pos[i])) - pos[0]
                    #if not positions[i] in positions[:i]:
                        #positions_unique.append(np.array(list(positions[i])) - positions_unique[0])
                positions.append(pos)
                line = f.readline()
            positions = np.array(positions)
        return positions

    def convert_to_relative(self,global_positions):
        #from LHip to LeftToeEndSite
        positions = np.copy(global_positions)
        for i in range(7):
            positions[:,7-i,:] = positions[:,7-i,:] - positions[:,7-(i+1),:]

        #from RHit to RightToeEndSite
        for i in range(5):
            positions[:,13-i,:] = positions[:,13-i,:] - positions[:,13-(i+1),:]
        positions[:,8,:] = np.zeros(3)#positions[:,8,:] - positions[:,0,:]

        #from LowerBack to HeadEndSite
        for i in range(6):
            positions[:,20-i,:] = positions[:,20-i,:] - positions[:,20-(i+1),:]
        positions[:,14,:] = np.zeros(3)#positions[:,14,:] - positions[:,0,:]

        #from LShoulder to LHandEndSite
        for i in range(6):
            positions[:,27-i,:] = positions[:,27-i,:] - positions[:,27-(i+1),:]
        positions[:,21,:] = np.zeros(3)#positions[:,16,:]

        #from LThunb to EndSite
        positions[:,29,:] = positions[:,29,:] - positions[:,28,:]
        positions[:,28,:] = np.zeros(3)#positions[:,24,:]

        #from RShoulder to RHandEndSite
        for i in range(6):
            positions[:,36-i,:] = positions[:,36-i,:] - positions[:,36-(i+1),:]
        positions[:,30,:] = np.zeros(3)#positions[:,16,:]

        #from RThunb to HeadEndSite
        positions[:,38,:] = positions[:,38,:] - positions[:,37,:]
        positions[:,37,:] = np.zeros(3)#positions[:,33,:]

        return positions

    def normalize_by_offset(self,positions, offsets):
        norm_positions = np.zeros(positions.shape)
        norm_positions[:,0,:] = positions[:,0,:]
        for i in range(1, positions.shape[1]):
            #offsetのnormが0のものを省く
            if np.sum(np.sum(np.abs(offsets[i-1,:]), axis=0), axis=0) > 0:
                offset_norm = np.linalg.norm(offsets[i-1,:])
                norm_positions[:,i,:] = positions[:,i,:] / offset_norm

            else:
                norm_positions[:,i,:] = np.zeros((positions.shape[0],3))
        #print(norm_positions[1,20,:])
        return norm_positions

    def denormalization_by_offset(self,norm_positions, offsets):
        positions = np.zeros(norm_positions.shape)
        positions[:,0,:] = norm_positions[:,0,:]
        for i in range(1, offsets.shape[0]+1):
            if np.sum(np.sum(np.abs(offsets[i-1,:]), axis=0), axis=0) > 0:
                offset_norm = np.linalg.norm(offsets[i-1,:])
                for j in range(positions.shape[0]):
                    norm = np.linalg.norm(norm_positions[j,i,:])
                    if norm > 1e-6:
                        positions[j,i,:] = (norm_positions[j,i,:] / norm) * offset_norm
                    else:
                        positions[j,i,:] = np.zeros(3)
            else:
                positions[:,i,:] = np.zeros((norm_positions.shape[0], 3))
        return positions

    def convert_to_global(self,relative_positions, root):
        positions = np.copy(relative_positions)
        positions[:,1,:] = root
        #from LHip to LeftToeEndSite
        for i in range(1,7):
            positions[:,i+1,:] = positions[:,i+1,:] + positions[:,i,:]

        #from RHit to RightToeEndSite
        positions[:,8,:] = positions[:,1,:]
        for i in range(5):
            positions[:,8+(i+1),:] = positions[:,8+(i+1),:] + positions[:,8+i,:]

        #from LowerBack to HeadEndSite
        positions[:,14,:] = positions[:,1,:]
        for i in range(6):
            positions[:,14+(i+1),:] = positions[:,14+(i+1),:] + positions[:,14+i,:]

        #from LShoulder to LHandEndSite
        positions[:,21,:] = positions[:,16,:]
        for i in range(6):
            positions[:,21+(i+1),:] = positions[:,21+(i+1),:] + positions[:,21+i,:]

        #from LThunb to EndSite
        positions[:,28,:] = positions[:,24,:]
        positions[:,29,:] = positions[:,29,:] + positions[:,28,:]

        #from RShoulder to RHandEndSite
        positions[:,30,:] = positions[:,16,:]
        for i in range(6):
            positions[:,30+(i+1),:] = positions[:,30+(i+1),:] + positions[:,30+i,:]

        #from RThunb to HeadEndSite
        positions[:,37,:] = positions[:,33,:]
        positions[:,38,:] = positions[:,38,:] + positions[:,37,:]

        return positions

    def zero_cut(self,positions, offsets):
        size = 0
        for j in range(offsets.shape[0]):
            if np.sum(np.sum(np.abs(offsets[j,:]), axis=0), axis=0) > 0:
                size += 1
        non_zero_positions = np.zeros((positions.shape[0], size+1, 3))
        non_zero_positions[:,0,:] = positions[:,0,:]
        count = 1
        for i in range(1, offsets.shape[0]+1):
            if np.sum(np.sum(np.abs(offsets[i-1,:]), axis=0), axis=0) > 0:
                non_zero_positions[:,count,:] = positions[:,i,:]
                count += 1
        return non_zero_positions

    def zero_add(self,non_zero_positions, offsets):
        positions = np.zeros((non_zero_positions.shape[0], offsets.shape[0]+1, 3))
        positions[:,0,:] = non_zero_positions[:,0,:]
        positions[:,1,:] = np.zeros((non_zero_positions.shape[0],3))
        count = 1
        for i in range(2, offsets.shape[0]+1):
            if np.sum(np.sum(np.abs(offsets[i-1,:]), axis=0), axis=0) > 0:
                positions[:,i,:] = non_zero_positions[:,count,:]
                count += 1
            else:
                positions[:,i,:] = positions[:,i-1,:]
        positions[:,[8,14,21,28,30,37],:] = positions[:,[1,1,16,24,16,33],:]
        return positions

def main():
    textlist = glob.glob('data/bvh/CMU_jp/Styled_jp/Subject137/*.txt')

    n = Normalize()

    for textpath in textlist:
        print(textpath)
        positions = n.readtxt(textpath)
        bvh_path = os.path.splitext(textpath)[0] + '.bvh'
        offsets = n.parse_initial_pos(bvh_path)
        relative_positions = n.convert_to_relative(positions)
        norm_positions = n.normalize_by_offset(relative_positions, offsets)
        root_norm_positions = n.convert_to_global(norm_positions, np.zeros((norm_positions.shape[0],3)))  #腰をルートとして、全関節の長さを１に正規化した座標系
        non_zero_root_norm_positions = n.zero_cut(root_norm_positions, offsets)
        np.save(os.path.splitext(textpath)[0] + 'n.npy', non_zero_root_norm_positions)
        root_norm_positions = n.zero_add(non_zero_root_norm_positions, offsets)
        rec_norm_positions = n.convert_to_relative(root_norm_positions)
        denorm_relative_positions = n.denormalization_by_offset(rec_norm_positions, offsets)
        global_positions = n.convert_to_global(denorm_relative_positions, denorm_relative_positions[:,0,:])
        name = os.path.split(os.path.splitext(textpath)[0])[1]
        print(global_positions.shape, positions[1,20,:],global_positions[1,20,:])

    npy_list = glob.glob('data/bvh/CMU_jp/Styled_jp/Subject137/*.npy')
    for npy_path in npy_list:
        npy = np.load(npy_path)
        top, name = os.path.split(npy_path)
        top, motion_name = os.path.split(top)

        if not os.path.exists(f'data/train_jp/CMU_jp/Styled_jp/Subject137'):
            os.makedirs(f'data/train_jp/CMU_jp/Styled_jp/Subject137')
        if not os.path.exists(f'data/test_jp/CMU_jp/Styled_jp/Subject137'):
            os.makedirs(f'data/test_jp/CMU_jp/Styled_jp/Subject137')

        length = npy.shape[0]
        train = npy[:int(length*0.9),:,:]
        np.save(f'data/train_jp/CMU_jp/Styled_jp/Subject137/{name}', train)
        test = npy[int(length*0.9):,:,:]
        np.save(f'data/test_jp/CMU_jp/Styled_jp/Subject137/{name}', npy)


if __name__ == '__main__':
    main()
