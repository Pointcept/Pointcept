import argparse
import os, sys

from SensorData import SensorData


def reader(
    filename,
    output_path,
    frame_skip,
    export_color_images=False,
    export_depth_images=False,
    export_poses=False,
    export_intrinsics=False,
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the data
    print("loading %s..." % filename)
    sd = SensorData(filename)
    if export_depth_images:
        sd.export_depth_images(
            os.path.join(output_path, "depth"), frame_skip=frame_skip
        )
    if export_color_images:
        sd.export_color_images(
            os.path.join(output_path, "color"), frame_skip=frame_skip
        )
    if export_poses:
        sd.export_poses(os.path.join(output_path, "pose"), frame_skip=frame_skip)
    if export_intrinsics:
        sd.export_intrinsics(os.path.join(output_path, "intrinsic"))
