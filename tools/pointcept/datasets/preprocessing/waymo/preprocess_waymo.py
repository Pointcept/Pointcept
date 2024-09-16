"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from pathlib import Path
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def create_lidar(frame):
    """Parse and save the lidar data in psd format.
    Args:
        frame (:obj:`Frame`): Open dataset frame proto.
    """
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points, valid_masks = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        keep_polar_features=True,
    )
    points_ri2, cp_points_ri2, valid_masks_ri2 = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1,
        keep_polar_features=True,
    )

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # point labels.

    points_all = np.concatenate([points_all, points_all_ri2], axis=0)

    velodyne = np.c_[points_all[:, 3:6], points_all[:, 1]]
    velodyne = velodyne.reshape((velodyne.shape[0] * velodyne.shape[1]))

    valid_masks = [valid_masks, valid_masks_ri2]
    return velodyne, valid_masks


def create_label(frame):
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels
    )
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels, ri_index=1
    )

    # point labels.
    point_labels_all = np.concatenate(point_labels, axis=0)
    point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
    point_labels_all = np.concatenate([point_labels_all, point_labels_all_ri2], axis=0)

    labels = point_labels_all
    return labels


def convert_range_image_to_cartesian(
    frame, range_images, range_image_top_pose, ri_index=0, keep_polar_features=False
):
    """Convert range images from polar coordinates to Cartesian coordinates.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.

    Returns:
      dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
        will be 3 if keep_polar_features is False (x, y, z) and 6 if
        keep_polar_features is True (range, intensity, elongation, x, y, z).
    """
    cartesian_range_images = {}
    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4])
    )

    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims,
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )

    for c in frame.context.laser_calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims
        )
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == open_dataset.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

        if keep_polar_features:
            # If we want to keep the polar coordinate features of range, intensity,
            # and elongation, concatenate them to be the initial dimensions of the
            # returned Cartesian range image.
            range_image_cartesian = tf.concat(
                [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1
            )

        cartesian_range_images[c.name] = range_image_cartesian

    return cartesian_range_images


def convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose,
    ri_index=0,
    keep_polar_features=False,
):
    """Convert range images to point cloud.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      camera_projections: A dict of {laser_name,
        [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.

    Returns:
      points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        (NOTE: Will be {[N, 6]} if keep_polar_features is true.
      cp_points: {[N, 6]} list of camera projections of length 5
        (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    valid_masks = []

    cartesian_range_images = convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features
    )

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims
        )
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(
            range_image_cartesian, tf.compat.v1.where(range_image_mask)
        )

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.compat.v1.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        valid_masks.append(range_image_mask.numpy())

    return points, cp_points, valid_masks


def convert_range_image_to_point_cloud_labels(
    frame, range_images, segmentation_labels, ri_index=0
):
    """Convert segmentation labels from range images to point clouds.

    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
    points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def handle_process(file_path, output_root, test_frame_list):
    file = os.path.basename(file_path)
    split = os.path.basename(os.path.dirname(file_path))
    print(f"Parsing {split}/{file}")
    save_path = Path(output_root) / split / file.split(".")[0]

    data_group = tf.data.TFRecordDataset(file_path, compression_type="")
    for data in data_group:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        context_name = frame.context.name
        timestamp = str(frame.timestamp_micros)

        if split != "testing":
            # for training and validation frame, extract labelled frame
            if not frame.lasers[0].ri_return1.segmentation_label_compressed:
                continue
        else:
            # for testing frame, extract frame in test_frame_list
            if f"{context_name},{timestamp}" not in test_frame_list:
                continue

        os.makedirs(save_path / timestamp, exist_ok=True)

        # extract frame pass above check
        point_cloud, valid_masks = create_lidar(frame)
        point_cloud = point_cloud.reshape(-1, 4)
        coord = point_cloud[:, :3]
        strength = np.tanh(point_cloud[:, -1].reshape([-1, 1]))
        pose = np.array(frame.pose.transform, np.float32).reshape(4, 4)
        mask = np.array(valid_masks, dtype=object)

        np.save(save_path / timestamp / "coord.npy", coord)
        np.save(save_path / timestamp / "strength.npy", strength)
        np.save(save_path / timestamp / "pose.npy", pose)

        # save mask for reverse prediction
        if split != "training":
            np.save(save_path / timestamp / "mask.npy", mask)

        # save label
        if split != "testing":
            # ignore TYPE_UNDEFINED, ignore_index 0 -> -1
            label = create_label(frame)[:, 1].reshape([-1]) - 1
            np.save(save_path / timestamp / "segment.npy", label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Waymo dataset",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--splits",
        required=True,
        nargs="+",
        choices=["training", "validation", "testing"],
        help="Splits need to process ([training, validation, testing]).",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    config = parser.parse_args()

    # load file list
    file_list = glob.glob(
        os.path.join(os.path.abspath(config.dataset_root), "*", "*.tfrecord")
    )
    assert len(file_list) == 1150

    # Create output directories
    for split in config.splits:
        os.makedirs(os.path.join(config.output_root, split), exist_ok=True)

    file_list = [
        file
        for file in file_list
        if os.path.basename(os.path.dirname(file)) in config.splits
    ]

    # Load test frame list
    test_frame_file = os.path.join(
        os.path.dirname(__file__), "3d_semseg_test_set_frames.txt"
    )
    test_frame_list = [x.rstrip() for x in (open(test_frame_file, "r").readlines())]

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            handle_process,
            file_list,
            repeat(config.output_root),
            repeat(test_frame_list),
        )
    )
