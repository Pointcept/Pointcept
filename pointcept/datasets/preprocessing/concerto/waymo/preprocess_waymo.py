"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cam_order_list = ["SIDE_RIGHT", "SIDE_LEFT", "FRONT_RIGHT", "FRONT_LEFT", "FRONT"]

import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from pathlib import Path
from PIL import Image
import io
import open3d as o3d

os.environ["OMP_NUM_THREADS"] = "1"
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
import glob
import multiprocessing as mp


def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4]
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            open_dataset.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()


def get_normals(cam_center, coords):
    Cs = np.repeat(cam_center.reshape((1, -1)), coords.shape[0], axis=0)
    view_dirs = coords - Cs
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    dot_product = np.sum(normals * view_dirs, axis=-1)
    flip_mask = dot_product > 0
    normals[flip_mask] = -normals[flip_mask]

    # Normalize normals a nd m
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals


def create_lidar_and_normals(frame):
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    all_points_vehicle_frame = []
    all_normals_vehicle_frame = []
    all_valid_masks = []

    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)

    for c in calibrations:
        points_0, _, valid_masks_0 = convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
            keep_polar_features=False,
        )

        points_1, _, valid_masks_1 = convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1,
            keep_polar_features=False,
        )

        sensor_index = calibrations.index(c)
        sensor_points_0 = points_0[sensor_index]
        sensor_points_1 = points_1[sensor_index]

        sensor_points_all = np.concatenate([sensor_points_0, sensor_points_1], axis=0)

        if sensor_points_all.shape[0] == 0:
            continue

        extrinsic = np.array(c.extrinsic.transform).reshape(4, 4)
        lidar_center_vehicle_frame = extrinsic[:3, 3]

        normals = get_normals(lidar_center_vehicle_frame, sensor_points_all)

        all_points_vehicle_frame.append(sensor_points_all)
        all_normals_vehicle_frame.append(normals)
        all_valid_masks.append(valid_masks_0[sensor_index])
        all_valid_masks.append(valid_masks_1[sensor_index])

    final_points = np.concatenate(all_points_vehicle_frame, axis=0)
    final_normals = np.concatenate(all_normals_vehicle_frame, axis=0)
    velodyne, _ = create_lidar(frame)
    strength = np.tanh(velodyne.reshape(-1, 4)[:, -1].reshape([-1, 1]))
    point_cloud_with_strength = np.c_[final_points, strength]
    return point_cloud_with_strength, final_normals, all_valid_masks


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


def project_lidar_to_image_with_color(
    vehicle_pose, lidar_points, image, calibration, lidar_colors
):
    """
    Projects LiDAR points to the image, fetches pixel color and pixel coordinates.
    Returns:
        filtered_points: (M, 3) - 3D points in camera frame that project onto the image.
        colors:          (M, 3) - RGB colors at projected 2D locations.
        uv_coords:       (M, 2) - Integer pixel coordinates (u, v) on the image.
        mask:            (N,)   - (optional) Boolean mask indicating which lidar points are used.
    """

    lidar_uv_coords = project_vehicle_to_image(vehicle_pose, calibration, lidar_points)
    u, v, ok = lidar_uv_coords.transpose()
    ok = ok.astype(bool)

    # Skip object if any corner projection failed. Note that this is very
    # strict and can lead to exclusion of some partially visible objects.

    # Clip box to image bounds.
    # u = np.clip(u.round(), 0, calibration.width-1).astype(int)
    # v = np.clip(v.round(), 0, calibration.height-1).astype(int)
    u_rounded = u.round().astype(int)
    v_rounded = v.round().astype(int)
    mask_u_valid = (u_rounded >= 0) & (u_rounded < calibration.width)
    mask_v_valid = (v_rounded >= 0) & (v_rounded < calibration.height)
    valid_mask = mask_u_valid & mask_v_valid & ok
    u_filtered = u_rounded[valid_mask]
    v_filtered = v_rounded[valid_mask]
    ok = ok & valid_mask
    lidar_colors[ok] = image[v_filtered, u_filtered, :]
    lidar_uv_coords = lidar_uv_coords[:, :2]
    return lidar_colors, lidar_uv_coords, valid_mask


def handle_process(file_path, output_root, test_frame_list):
    file = os.path.basename(file_path)
    split = os.path.basename(os.path.dirname(file_path))
    print(f"Parsing {split}/{file}")
    save_path = Path(output_root) / split / file.split(".")[0]

    data_group = tf.data.TFRecordDataset(file_path, compression_type="")
    for data in data_group:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        context_name = frame.context.name
        timestamp = str(frame.timestamp_micros)
        img_save_path = (
            Path(output_root) / "images" / split / file.split(".")[0] / timestamp
        )
        color_save_path = img_save_path / "color"
        correspondence_save_path = img_save_path / "correspondence"
        intrinsic_save_path = img_save_path / "intrinsic"
        pose_save_path = img_save_path / "pose"

        if split != "testing":
            # for training and validation frame, extract labelled frame
            if not frame.lasers[0].ri_return1.segmentation_label_compressed:
                continue
        else:
            # for testing frame, extract frame in test_frame_list
            if f"{context_name},{timestamp}" not in test_frame_list:
                continue
        os.makedirs(color_save_path, exist_ok=True)
        os.makedirs(correspondence_save_path, exist_ok=True)
        os.makedirs(intrinsic_save_path, exist_ok=True)
        os.makedirs(pose_save_path, exist_ok=True)
        os.makedirs(save_path / timestamp, exist_ok=True)

        # extract frame pass above check
        point_cloud, normal, valid_masks = create_lidar_and_normals(frame)
        point_cloud = point_cloud.reshape(-1, 4)
        coord = point_cloud[:, :3]
        strength = np.tanh(point_cloud[:, -1].reshape([-1, 1]))
        pose = np.array(frame.pose.transform, np.float32).reshape(4, 4)
        mask = np.array(valid_masks, dtype=object)

        np.save(save_path / timestamp / "coord.npy", coord)
        np.save(save_path / timestamp / "strength.npy", strength)
        np.save(save_path / timestamp / "pose.npy", pose)
        np.save(save_path / timestamp / "normal.npy", normal)

        # save mask for reverse prediction
        if split != "training":
            np.save(save_path / timestamp / "mask.npy", mask)

        # save label
        if split != "testing":
            # ignore TYPE_UNDEFINED, ignore_index 0 -> -1
            label = create_label(frame)[:, 1].reshape([-1]) - 1
            np.save(save_path / timestamp / "segment.npy", label)

        img_DLC = {}
        for image in frame.images:
            camera_name = open_dataset.CameraName.Name.Name(image.name)
            img_DLC[camera_name] = np.array(Image.open(io.BytesIO(image.image)))
            filename = color_save_path / f"{camera_name}.jpg"
            with open(filename, "wb") as f:
                f.write(image.image)
        color = np.zeros((coord.shape[0], 3))
        #    {'FRONT': 0, 'SIDE_RIGHT': 1, 'SIDE_LEFT': 2, ...}
        order_map = {name: i for i, name in enumerate(cam_order_list)}
        cam_sorted_calibrations = sorted(
            frame.context.camera_calibrations,
            key=lambda calib: order_map.get(
                open_dataset.CameraName.Name.Name(calib.name), 999
            ),
        )
        valid_mask = np.full((coord.shape[0],), False, dtype=bool)
        for c in cam_sorted_calibrations:
            camera_name = open_dataset.CameraName.Name.Name(c.name)
            cam2ego = np.array(c.extrinsic.transform).reshape(4, 4)
            intrinsic = np.array(c.intrinsic)
            fx = intrinsic[0]
            fy = intrinsic[1]
            cx = intrinsic[2]
            cy = intrinsic[3]
            intrinsic = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )
            np.save(intrinsic_save_path / f"{camera_name}.npy", intrinsic)
            np.save(pose_save_path / f"{camera_name}.npy", cam2ego)
            color, uv_correspondence, mask = project_lidar_to_image_with_color(
                frame.pose, coord, img_DLC[camera_name], c, color
            )
            valid_mask = np.logical_or(valid_mask, mask)
            correspondence_point_id = np.array(
                range(uv_correspondence.shape[0])
            ).reshape((-1, 1))
            uv_correspondence = np.hstack([uv_correspondence, correspondence_point_id])
            np.save(correspondence_save_path / f"{camera_name}.npy", uv_correspondence)
        np.save(save_path / timestamp / "color.npy", color)
        np.save(save_path / timestamp / "mask.npy", valid_mask)


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
    parser.add_argument(
        "--thread_id",
        default=0,
        type=int,
        help="Thread id.",
    )
    config = parser.parse_args()

    # load file list
    origin_splits = ["training", "validation", "testing"]
    file_list = []
    for split in origin_splits:
        file_list.extend(
            glob.glob(
                os.path.join(os.path.abspath(config.dataset_root), split, "*.tfrecord")
            )
        )
    assert len(file_list) == 1150, f"have {len(file_list)} files"

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
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "waymo/3d_semseg_test_set_frames.txt",
    )
    test_frame_list = [x.rstrip() for x in (open(test_frame_file, "r").readlines())]

    split_file_list = np.array_split(file_list, config.num_workers)
    split_file_list_ = split_file_list[config.thread_id]
    for file_list_i in split_file_list_:
        handle_process(file_list_i, config.output_root, test_frame_list)
