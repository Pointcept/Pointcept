"""
Preprocessing Script for nuScenes Informantion
modified from OpenPCDet (https://github.com/open-mmlab/OpenPCDet)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
from pathlib import Path
import numpy as np
import argparse
import tqdm
import pickle
from functools import reduce
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix


map_name_from_general_to_detection = {
    "human.pedestrian": "pedestrian",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}


def get_available_scenes(nusc):
    available_scenes = []
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"][nusc.lidar_chan])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    return available_scenes


def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor"s coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    else:
        cam_intrinsic = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: "lidar".

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    # if os.getcwd() in data_path:  # path from lyftdataset is absolute path
    #     data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)
    sweep[f"sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def fill_trainval_infos(
    data_path, nusc, train_scenes, radar_chan, ref_chan,test=False, max_sweeps=10, with_camera=False,  with_lidar=True, map_to_lidar = True,
):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(
        total=len(nusc.sample), desc="create_info", dynamic_ncols=True
    )

    ref_chan = ref_chan  # The reference channel from which we track back n sweeps to aggregate the point cloud.
    chan = radar_chan  # The radar channel of the current sample_rec that the point clouds are mapped to.
    lidar_chan = nusc.lidar_chan
    for index, sample in enumerate(nusc.sample):
        progress_bar.update()
        camera_types = nusc.get_camera_types()
        available_cameras = [key for key in camera_types if key in sample["data"]]
        if nusc.cam_chan not in available_cameras:
            continue
        if nusc.lidar_chan not in sample["data"] and with_lidar:
            continue
        ref_lidar_token = sample["data"].get(lidar_chan)
        ref_lidar_rec = nusc.get("sample_data", ref_lidar_token)
        ref_lidar_cs_rec = nusc.get(
            "calibrated_sensor", ref_lidar_rec["calibrated_sensor_token"]
        )
        ref_lidar_pose_rec = nusc.get("ego_pose", ref_lidar_rec["ego_pose_token"])

        ref_sd_token = sample["data"].get(ref_chan, None)
        if ref_sd_token is None:
            continue
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        assert ref_sd_rec['sensor_modality'] == "radar", f"reference channel should be radar, but got {ref_sd_rec['sensor_modality']}"

        ref_cs_rec = nusc.get(
            "calibrated_sensor", ref_sd_rec["calibrated_sensor_token"]
        )
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        ref_radar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)
        ref_lidar_path, _,_ = get_sample_data(nusc, ref_lidar_token)
        ref_cam_front_token = sample["data"][nusc.cam_chan]
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )
        # Get lidar pose
        global_from_car = transform_matrix(
            ref_lidar_pose_rec["translation"],
            Quaternion(ref_lidar_pose_rec["rotation"]),
            inverse=False,
        )

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        car_from_lidar = transform_matrix(
            ref_lidar_cs_rec["translation"],
            Quaternion(ref_lidar_cs_rec["rotation"]),
            inverse=False,
        )
        tm = reduce(
            np.dot,
            [ref_from_car, car_from_global, global_from_car, car_from_lidar],
        )
        info = {
            "radar_path": Path(ref_radar_path).relative_to(data_path).__str__(),
            "lidar_path": Path(ref_lidar_path).relative_to(data_path).__str__(),
            "lidar_to_ref": tm,
            "radar_token": ref_sd_token,
            "cam_front_path": Path(ref_cam_path).relative_to(data_path).__str__(),
            "cam_intrinsic": ref_cam_intrinsic,
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
        }
        if with_camera:
            info["cams"] = dict()
            l2e_r = ref_lidar_cs_rec["rotation"] if map_to_lidar else ref_cs_rec["rotation"]
            l2e_t = (ref_lidar_cs_rec["translation"],) if map_to_lidar else (ref_cs_rec["translation"],)
            e2g_r = ref_lidar_pose_rec["rotation"] if map_to_lidar else ref_pose_rec["rotation"]
            e2g_t = ref_lidar_pose_rec["translation"] if map_to_lidar else ref_pose_rec["translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame 
        # obtain 6 image's information per frame

            for cam in available_cameras:
                cam_token = sample["data"][cam]
                cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info["data_path"] = (
                    Path(cam_info["data_path"]).relative_to(data_path).__str__()
                )
                cam_info.update(camera_intrinsics=camera_intrinsics)
                info["cams"].update({cam: cam_info})

        sample_data_token = sample["data"].get(chan, None)
        if sample_data_token is None:
            continue
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec["prev"] == "":
                if len(sweeps) == 0:
                    sweep = {
                        "radar_path": Path(ref_radar_path)
                        .relative_to(data_path)
                        .__str__(),
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": None,
                        "time_lag": curr_sd_rec["timestamp"] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

                # Get past pose
                current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
                global_from_car = transform_matrix(
                    current_pose_rec["translation"],
                    Quaternion(current_pose_rec["rotation"]),
                    inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"]
                )
                car_from_current = transform_matrix(
                    current_cs_rec["translation"],
                    Quaternion(current_cs_rec["rotation"]),
                    inverse=False,
                )

                tm = reduce(
                    np.dot,
                    [ref_from_car, car_from_global, global_from_car, car_from_current],
                )

                radar_path = nusc.get_sample_data_path(curr_sd_rec["token"])

                time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                sweep = {
                    "radar_path": Path(radar_path).relative_to(data_path).__str__(),
                    "sample_data_token": curr_sd_rec["token"],
                    "transform_matrix": tm,
                    "global_from_car": global_from_car,
                    "car_from_current": car_from_current,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        assert len(info["sweeps"]) == max_sweeps - 1, (
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, "
            f"you should duplicate to sweep num {max_sweeps - 1}"
        )

        if not test:
            # processing gt bbox
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array([anno["num_lidar_pts"] for anno in annotations])
            num_radar_pts = np.array([anno[f"num_{radar_chan}_pts"] for anno in annotations])
            mask = num_lidar_pts + num_radar_pts > 0

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[
                :, [1, 0, 2]
            ]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(
                -1, 1
            )
            rotation_matrices = np.array([b.rotation_matrix for b in ref_boxes]).reshape(-1, 3, 3)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)
            
            assert len(annotations) == len(gt_boxes) == len(velocity)

            info["gt_boxes"] = gt_boxes[mask, :]
            info["gt_boxes_velocity"] = velocity[mask, :]
            info["gt_boxes_rotation_matrices"] = rotation_matrices[mask]
            info["gt_names"] = np.array(
                [map_name_from_general_to_detection[name] for name in names]
            )[mask]
            info["gt_boxes_token"] = tokens[mask]
            info["num_lidar_pts"] = num_lidar_pts[mask]
            info["num_radar_pts"] = num_radar_pts[mask]


        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Path to the nuScenes dataset."
    )
    parser.add_argument(
        "--dataset_version", required=True, help="Version of the nuScenes dataset."
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where processed information located.",
    )
    parser.add_argument(
        "--max_sweeps", default=10, type=int, help="Max number of sweeps. Default: 10."
    )
    parser.add_argument(
        "--with_camera",
        action="store_true",
        default=False,
        help="Whether use camera or not.",
    )
    parser.add_argument(
        "--radar_chan",
        required=True,
        help="Which radar channel to use.",
    )
    parser.add_argument(
        "--ref_chan",
        required=True,
        help="Which reference channel to use.",
    )
    config = parser.parse_args()

    print(f"Loading nuScenes tables for version {config.dataset_version}...")
    nusc_trainval = NuScenes(
        version=config.dataset_version, dataroot=config.dataset_root, verbose=True
    )
    available_scenes_trainval = get_available_scenes(nusc_trainval)
    available_scene_names_trainval = [s["name"] for s in available_scenes_trainval]
    print("total scene num:", len(nusc_trainval.scene))
    print("exist scene num:", len(available_scenes_trainval))

    split_scenes = nusc_trainval.create_splits_scenes()
    train_scenes = split_scenes["train"]
    val_scenes = split_scenes["val"]
    test_scenes = split_scenes["test"]
    train_scenes = set(
        [
            available_scenes_trainval[available_scene_names_trainval.index(s)]["token"]
            for s in train_scenes
        ]
    )

    print(f"Filling trainval information...")
    train_nusc_infos, val_nusc_infos = fill_trainval_infos(
        config.dataset_root,
        nusc_trainval,
        train_scenes,
        radar_chan=config.radar_chan,
        ref_chan=config.ref_chan,
        test=False,
        max_sweeps=config.max_sweeps,
        with_camera=config.with_camera,
    )
   
    print(f"Saving nuScenes information...")
    os.makedirs(os.path.join(config.output_root, "info"), exist_ok=True)
    print(
        f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
    )
    with open(
        os.path.join(
            config.output_root,
            "info",
            f"nuscenes_infos_{config.max_sweeps}sweeps_train.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(train_nusc_infos, f)
    with open(
        os.path.join(
            config.output_root,
            "info",
            f"nuscenes_infos_{config.max_sweeps}sweeps_val.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(val_nusc_infos, f)
    print("Done!")