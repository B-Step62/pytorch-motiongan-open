import os
import argparse

from core.datasets.create_joint_dataset import collect_bvh, convert_bvh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--standard_bvh', default=None, required=False)
    args = parser.parse_args()

    bvh_root = args.dataset
    out = args.out if args.out is not None else bvh_root

    # collect bvh paths
    out_dir, bvh_paths = collect_bvh(bvh_root, out)
    standard_bvh = args.standard_bvh if args.standard_bvh is not None else bvh_paths[0]

    convert_bvh(bvh_paths, out_dir, standard_bvh)


if __name__ == '__main__':
    main()
