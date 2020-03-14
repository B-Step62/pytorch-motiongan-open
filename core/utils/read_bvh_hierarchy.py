import re
from collections import OrderedDict


# Bone has parent, chennels, offsets
def new_bone(parent, name):
    bone = {'parent':parent, 'children':[], 'channels':[], 'offsets':[]}
    return bone

# Scanner to parse bvh
def create_scanner():
    scanner = re.Scanner([
            (r'[a-zA-Z_]\w*', lambda sc, tk : ('IDENT', tk)),
            (r'-*[0-9]+(\.[0-9]+)?', lambda sc, tk : ('DIGIT', tk)),
            (r'/-?(?:0|[1-9]\d*)(?:[eE][+\-]?\d+)?/', lambda sc, tk : ('DIGIT', tk)),
            (r'{', lambda sc, tk : ('OPEN_BRACE', tk)),
            (r'}', lambda sc, tk : ('CLOSE_BRACE', tk)), 
            (r':', None),
            (r'\s+', None),
            ])
    return scanner



# Read offsets from tokens
def read_offset(bvh_tokens, token_index):
    # Offset start with IDENT 'OFFSET'
    assert bvh_tokens[token_index] == ('IDENT', 'OFFSET')
    token_index += 1
    offsets = list(map(lambda tk: float(tk[1]), bvh_tokens[token_index:token_index+3])) 
    return offsets, token_index+3 



# Read channels from tokens
def read_channels(bvh_tokens, token_index):
    # Channels start with IDENT 'CHANNELS'
    assert bvh_tokens[token_index] == ('IDENT', 'CHANNELS')
    token_index += 1
    channel_count = int(bvh_tokens[token_index][1])
    token_index += 1
    channels = list(map(lambda tk : tk[1], bvh_tokens[token_index:token_index+channel_count]))
    return channels, token_index+channel_count



# Parse each joint
def parse_joint(bvh_tokens, token_index, skelton, current_tree_path, non_end_bones, motion_channels, permute_xyz_order):
    joint_id = bvh_tokens[token_index][1] # JOINT or END
    token_index += 1
    joint_name = bvh_tokens[token_index][1]
    token_index += 1
    # If end site, set unique name
    if joint_id == 'End':
        joint_name = current_tree_path[-1] + '_EndSite'

    joint = new_bone(current_tree_path[-1], joint_name)
    # Joint start with open bracket
    assert bvh_tokens[token_index][0] == 'OPEN_BRACE' 
    token_index += 1
    offsets, token_index= read_offset(bvh_tokens, token_index)
    joint['offsets'] = offsets
    if joint_id == 'JOINT':
        channels, token_index = read_channels(bvh_tokens, token_index)
        # Rotate xyz order of rotation to zyx
        if channels.index('Xrotation') > -1:
            channels = [channels[p] for p in permute_xyz_order]
        joint['channels'] = channels
        
        non_end_bones.append(joint_name)
        for channel in channels:
            motion_channels.append((joint_name, channels))
    skelton[joint_name] = joint

    # Scan all children 
    while bvh_tokens[token_index] in [('IDENT', 'JOINT'), ('IDENT', 'End')]:
        current_tree_path.append(joint_name)
        child_name, token_index, skelton, current_tree_path, non_end_bones, motion_channels = parse_joint(bvh_tokens, token_index, skelton, current_tree_path, non_end_bones, motion_channels, permute_xyz_order)
        current_tree_path.pop()
        skelton[joint_name]['children'].append(child_name)

    # Joint tree end with close bracket
    assert bvh_tokens[token_index][0] == 'CLOSE_BRACE'
    return joint_name, token_index+1, skelton, current_tree_path, non_end_bones, motion_channels
   



# Parse hierarchy from scanned tokens
def parse_hierarchy(bvh_tokens, permute_xyz_order):
    # Start with "HIERARCHY \n ROOT [IDENT] {"
    assert bvh_tokens[0]==('IDENT','HIERARCHY') and bvh_tokens[1]==('IDENT','ROOT') and bvh_tokens[2][0]=='IDENT' and bvh_tokens[3][0]=='OPEN_BRACE'

    # Get root bone
    root_name = 'Root' #bvh_tokens[2][1]
    root_bone = new_bone(None, root_name)

    # Hierarchy start 
    token_index = 4 
    root_offsets, token_index = read_offset(bvh_tokens, token_index) 
    root_channels, token_index = read_channels(bvh_tokens, token_index)
    root_bone['offsets'] = [root_offsets[p] for p in permute_xyz_order]
    root_bone['channels'] = [root_channels[p] for p in permute_xyz_order]


    # Create skelton
    skelton = OrderedDict()
    skelton[root_name] = root_bone
    current_tree_path = []  # Store current path from root bone 
    motion_channels = []    # List of channel name written in motion part
    non_end_bones = []      # List of bones except End Site
    current_tree_path = [root_name]

    while bvh_tokens[token_index][1] == 'JOINT':
        child_name, token_index, skelton, current_tree_path, non_end_bones, motion_channels = parse_joint(bvh_tokens, token_index, skelton, current_tree_path, non_end_bones, motion_channels, permute_xyz_order)
        skelton[root_name]['children'].append(child_name)
    return skelton, non_end_bones

# Read xyz order in euler angle
def read_xyz_order(bvh_path):
    with open(bvh_path, 'r') as f:
        bvh = f.readlines()
    terms = bvh[4].split()
    x = terms.index('Xrotation')
    y = terms.index('Yrotation')
    z = terms.index('Zrotation')
    off = min(x,y,z)

    permute_xyz_order = [z-off, y-off, x-off]
    return permute_xyz_order


# Read hierarchy
def read_bvh_hierarchy(bvh_path):
    with open(bvh_path, 'r') as f:
        bvh = f.read()
    bvh_scanner = create_scanner()
    bvh_tokens, _ = bvh_scanner.scan(bvh)
    permute_xyz_order = read_xyz_order(bvh_path)
    skelton, non_end_bones = parse_hierarchy(bvh_tokens, permute_xyz_order)
    return skelton, non_end_bones, permute_xyz_order



def debug(bvh_path):
    bvh_path = '../../motionGAN/data/bvh/CMU_jp/Locomotion_jp/walking_jp/02_02.bvh'
    read_bvh_hierarchy(bvh_path)

if __name__ == '__main__':
    debug(bvh_path) 
