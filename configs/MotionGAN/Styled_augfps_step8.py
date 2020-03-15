# Models
models = dict(
    generator = dict(
        model = 'MotionGAN_generator',
        top = 64,
        padding_mode = 'reflect',
        kw = 5,
        w_dim = 64,
        use_z = 'transform',
        z_dim = 64,
        normalize_z = True,
    ),
    discriminator = dict(
        model = 'MotionGAN_discriminator',
        top = 64,
        padding_mode = 'reflect',
        kw = 5,
        norm = 'spectral',
        use_sigmoid = True,
    ),
)

# Traiing strategy
train = dict(
    batchsize = 24,
    num_workers = 16,
    total_iterations = 200000,
    out = 'results/MotionGAN/Styled_augfps_step8',

    # Dataset
    dataset=dict(
        data_root = './data/train_jp/CMU_jp_new/Styled_jp/Walk_jp',
        class_list = ['Cat', 'Chicken', 'Dinosaur', 'Drunk', 'GanglyTeen', 'GracefulLady', 'Normal', 'OldMan','SexyLady', 'StrongMan','Childish', 'Clumsy', 'Cool', 'Depressed', 'Elated', 'Elderlyman', 'Happy', 'Joy', 'Lavish', 'Marching', 'Painfulleftknee', 'Relaxed', 'Rushed', 'Sad', 'Scared', 'Sexy', 'Shy', 'Sneaky'],
        start_offset = 1,
        control_point_interval = 256,
        standard_bvh = 'core/datasets/CMU_standard.bvh',
        scale = 1.,
        frame_nums = 1024,
        frame_step = 8,
        augment_fps = True,
        rotate = True,
    ),

    # Iteration intervals
    display_interval = 100,
    preview_interval = 5000,
    save_interval = 20000,

    # Loss
    GAN_type = 'normal',
    trjloss_sampling_points = 4,
    parameters=dict(
        g_lr = 0.0002,
        d_lr = 0.0001,
        lam_g_adv = 1.,
        lam_g_trj = 0.1,
        lam_g_cls = 5.,
        lam_g_bone = 0.1,
        lam_d_adv = 1.,
        lam_d_cls = 5.,
    ),

    # Preview video parameters
    preview=dict(
        view_range = 50,
        save_delay = 4,
    ),
)

# Testing strategy
test = dict(
    out = 'results/MotionGAN/Styled_augfps_step8',
    dataset=dict(
        data_root = './data/test_jp/CMU_jp_new/Locomotion_jp/walking_jp',
        class_list = [],
        start_offset = 1,
        control_point_interval = 256,
        standard_bvh = 'core/datasets/CMU_standard.bvh',
        scale = 1.,
        frame_step = 8,
    ),
    preview=dict(
        view_range = 50,
        save_delay = 4,
    ),
)

