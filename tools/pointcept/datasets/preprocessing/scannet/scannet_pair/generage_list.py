# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob, os, sys

from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument("--target_dir", required=True, help="path to the target dir")

opt = parser.parse_args()
print(opt)


def main():
    overlaps = glob.glob(os.path.join(opt.target_dir, "*/pcd/overlap.txt"))
    with open(os.path.join(opt.target_dir, "overlap30.txt"), "w") as f:
        for fo in overlaps:
            for line in open(fo):
                pcd0, pcd1, op = line.strip().split()
                if float(op) >= 0.3:
                    print("{} {} {}".format(pcd0, pcd1, op), file=f)
    print("done")


if __name__ == "__main__":
    main()
