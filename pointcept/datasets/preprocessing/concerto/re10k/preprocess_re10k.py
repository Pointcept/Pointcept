"""
Preprocessing Script for RE10K using VGGT(https://github.com/facebookresearch/vggt)

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pathlib import Path
from torchvision.utils import save_image
import torch
from io import BytesIO
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as tf
import argparse
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from torchvision.utils import save_image

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import camtools as ct
from torchvision.transforms import transforms as T

_alignment = 14
target_size = 518

generator = torch.Generator()
generator.manual_seed(123)


def extract_and_align_ground_plane(
    pcd,
    height_percentile=20,
    ransac_distance_threshold=0.01,
    ransac_n=3,
    ransac_iterations=1000,
    max_angle_degree=40,
    max_trials=6,
):
    points = np.asarray(pcd.points)
    z_vals = points[:, 2]
    z_thresh = np.percentile(z_vals, height_percentile)
    low_indices = np.where(z_vals <= z_thresh)[0]

    remaining_indices = low_indices.copy()

    for trial in range(max_trials):
        if len(remaining_indices) < ransac_n:
            raise ValueError("Not enough points left to fit a plane.")

        low_pcd = pcd.select_by_index(remaining_indices)

        plane_model, inliers = low_pcd.segment_plane(
            distance_threshold=ransac_distance_threshold,
            ransac_n=ransac_n,
            num_iterations=ransac_iterations,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)

        angle = np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0)) * 180 / np.pi
        if angle <= max_angle_degree:
            inliers_global = remaining_indices[inliers]

            target = np.array([0, 0, 1])
            axis = np.cross(normal, target)
            axis_norm = np.linalg.norm(axis)

            if axis_norm < 1e-6:
                rotation_matrix = np.eye(3)
            else:
                axis /= axis_norm
                rot_angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
                rotation = R.from_rotvec(axis * rot_angle)
                rotation_matrix = rotation.as_matrix()

            rotated_points = points @ rotation_matrix.T
            ground_points_z = rotated_points[inliers_global, 2]
            offset = np.mean(ground_points_z)
            rotated_points[:, 2] -= offset

            aligned_pcd = o3d.geometry.PointCloud()
            aligned_pcd.points = o3d.utility.Vector3dVector(rotated_points)
            if pcd.has_colors():
                aligned_pcd.colors = pcd.colors
            if pcd.has_normals():
                rotated_normals = np.asarray(pcd.normals) @ rotation_matrix.T
                aligned_pcd.normals = o3d.utility.Vector3dVector(rotated_normals)

            return aligned_pcd, inliers_global, rotation_matrix, offset

        else:
            rejected_indices = remaining_indices[inliers]
            remaining_indices = np.setdiff1d(remaining_indices, rejected_indices)

    raise ValueError("Failed to find a valid ground plane within max trials.")


def rotx(x, theta=90):
    """
    Rotate x by theta degrees around the x-axis
    """
    theta = np.deg2rad(theta)
    rot_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    return rot_matrix @ x


def Coord2zup(points, extrinsics):
    """
    Convert the dust3r coordinate system to the z-up coordinate system
    """
    points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1).T
    points = rotx(points, -90)[:3].T
    t = np.min(points, axis=0)
    points -= t
    extrinsics = rotx(extrinsics, -90)
    extrinsics[:, :3, 3] -= t.T
    return points, extrinsics


def resize_images_intrinsic(images, intrinsics, size, crop_size):
    h, w = size
    crop_h, crop_w = crop_size
    intrinsics[:, 0, 0] = crop_w
    intrinsics[:, 1, 1] = crop_h
    cam_trans = torch.tensor(
        [[w / crop_w, 0, 0], [0, h / crop_h, 0], [0, 0, 1]], dtype=torch.float32
    )
    intrinsics = torch.stack([cam_trans @ intrinsics_i for intrinsics_i in intrinsics])
    return images, intrinsics


def calDelta(ang1, ang2, dist1, dist2):
    alpha = 20
    angD = ang1 - ang2
    distD = dist1 - dist2
    ang_s = np.linalg.norm(angD)
    dist_s = np.linalg.norm(distD)
    return ang_s + alpha * dist_s, ang_s, dist_s


def convert_poses(poses):
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def convert_images(images):
    torch_images = []
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes()))
        torch_images.append(tf.ToTensor()(image))
    return torch.stack(torch_images)


def plyGet(
    images,
    scene_dir,
    pc_outputdir,
    im_outputdir,
    device,
    dtype,
    vggt_model,
    depth_points="points",
    conf=0.0,
    parse_depths=False,
):
    # T0 = T0.numpy()
    print(f"Processing scene: {scene_dir}")
    height, width = images.shape[-2:]

    # Skip every n frames
    print(f"Processing {images.shape[0]} images")

    image_num = images.shape[0]
    # Load and preprocess images
    images = images.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = vggt_model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    Ts, Ks = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    Ts = Ts[0].detach().cpu().numpy()
    Ks = Ks[0].detach().cpu().numpy()
    Ts = ct.convert.pad_0001(Ts)
    Ts_inv = np.linalg.inv(Ts)
    Cs = np.array([ct.convert.T_to_C(T) for T in Ts])  # (n, 3)

    # correspondance
    pixel = np.transpose(np.indices((width, height)), (2, 1, 0))
    pixel = pixel.reshape((-1, 2))
    pixel_ = []
    for i in range(image_num):
        pixel_id = np.hstack((pixel, i * np.ones((pixel.shape[0], 1))))
        pixel_.append(pixel_id)
    pixel = np.concatenate(pixel_, axis=0)

    if depth_points == "points":
        world_points = predictions["world_points"].detach().cpu().numpy()
        world_points_conf = predictions["world_points_conf"].detach().cpu().numpy()
        img_depths = predictions["depth"].detach().cpu().numpy().squeeze(-1)
    else:
        img_depths = predictions["depth"].detach().cpu().numpy().squeeze(-1)
        img_depths_conf = predictions["depth_conf"].detach().cpu().numpy()
        world_points = [
            ct.project.im_depth_to_point_cloud(img_depth, K, T)
            for img_depth, K, T in zip(img_depths[0], Ks, Ts)
        ]
        world_points = np.stack(world_points).reshape(
            1, image_num, images.shape[-2], images.shape[-1], -1
        )
        world_points_conf = img_depths_conf

    masks = world_points_conf > conf
    points_masks = masks.reshape((image_num * images.shape[-2] * images.shape[-1],))

    # Compute view direction for each pixel
    # (b n h w c) - (n, 3)
    view_dirs = world_points - rearrange(Cs, "n c -> 1 n 1 1 c")
    view_dirs = rearrange(view_dirs, "b n h w c -> (b n h w) c")
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)

    # Extract points and colors
    # [1, 8, 3, 294, 518]
    images = images.detach().cpu().unsqueeze(0).numpy()
    points = rearrange(world_points, "b n h w c -> (b n h w) c")
    colors = rearrange(images, "b n c h w -> (b n h w) c")

    # masks
    points = points[points_masks]
    colors = colors[points_masks]
    pixel = pixel[points_masks]
    pixel = np.concatenate((pixel, np.arange(points.shape[0]).reshape(-1, 1)), axis=-1)

    points, Ts_inv = Coord2zup(points, Ts_inv)
    scale = 3 / (points[:, 2].max() - points[:, 2].min())
    points *= scale
    Ts_inv[:, :3, 3] *= scale

    # Create a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    try:
        pcd, inliers, rotation_matrix, offset = extract_and_align_ground_plane(pcd)
    except:
        print("cannot find ground")
        return
    T_pcd = np.eye(4)
    T_pcd[:3, :3] = rotation_matrix
    T_pcd[2, 3] = -offset
    Ts_inv = T_pcd @ Ts_inv
    os.makedirs(pc_outputdir, exist_ok=True)
    os.makedirs(im_outputdir, exist_ok=True)

    # Filp normals such that normals always point to camera
    # Compute the dot product between the normal and the view direction
    # If the dot product is less than 0, flip the normal
    normals = np.asarray(pcd.normals)
    view_dirs = np.asarray(view_dirs)
    view_dirs = view_dirs[points_masks]
    dot_product = np.sum(normals * view_dirs, axis=-1)
    flip_mask = dot_product > 0
    normals[flip_mask] = -normals[flip_mask]

    # Normalize normals a nd m
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    colors = (colors * 255).astype(np.uint8)

    np.save(Path(pc_outputdir) / "coord.npy", points)
    np.save(Path(pc_outputdir) / "color.npy", colors)
    np.save(Path(pc_outputdir) / "normal.npy", normals)

    correspondence_path = os.path.join(im_outputdir, "correspondence")
    img_output_path = os.path.join(im_outputdir, "color")
    Ts_output_path = os.path.join(im_outputdir, "pose")
    Ks_output_path = os.path.join(im_outputdir, "intrinsic")
    os.makedirs(img_output_path, exist_ok=True)
    os.makedirs(Ts_output_path, exist_ok=True)
    os.makedirs(Ks_output_path, exist_ok=True)
    os.makedirs(correspondence_path, exist_ok=True)
    for id in range(image_num):
        pixels_correspondence = pixel[pixel[:, 2] == id]
        pixels_correspondence = pixels_correspondence[:, [0, 1, 3]]
        np.save(
            os.path.join(im_outputdir, "correspondence", f"{id}.npy"),
            pixels_correspondence,
        )
    images = torch.tensor(images.squeeze(0))
    if parse_depths:
        depth_output_path = os.path.join(im_outputdir, "depth")
        os.makedirs(depth_output_path, exist_ok=True)
        img_depths = img_depths.squeeze(0)[..., np.newaxis] * scale * 1000
    for id in range(images.shape[0]):
        save_image(images[id], os.path.join(img_output_path, f"{id}.png"))
        np.save(os.path.join(Ts_output_path, f"{id}.npy"), Ts_inv[id])
        np.save(os.path.join(Ks_output_path, f"{id}.npy"), Ks[id])
        if parse_depths:
            cv2.imwrite(
                os.path.join(depth_output_path, f"{id}.png"),
                img_depths[id].astype(np.uint16),
            )


def parse_scene(
    chunk_path,
    vggt_model,
    num_context_views,
    outputdir,
    device,
    frame_gap,
    overlap_range,
    dtype,
    conf,
    parse_depths,
):
    print("loading from ", chunk_path)
    split = chunk_path.parts[-2]
    # Load the chunk.
    chunk = torch.load(chunk_path)
    for run_idx in range(len(chunk)):
        example = chunk[run_idx]
        extrinsics, intrinsics = convert_poses(example["cameras"])
        scene = example["key"]

        # Load the images.
        context_images = example["images"]
        context_images = convert_images(context_images)

        v, _, _ = extrinsics.shape
        ang_list = np.array(
            [
                R.from_matrix(ex_i[:3, :3].cpu().numpy()).as_euler("xyz", degrees=True)
                for ex_i in extrinsics
            ]
        )
        dist_list = np.array(extrinsics[:, :3, 3].reshape((-1, 3)).cpu())
        context_indices = torch.randperm(v, generator=generator)
        for context_index in tqdm(context_indices, "Finding context pair"):
            choose_index = [context_index.item()]
            while len(choose_index) < num_context_views:
                # Step away from context view until the minimum overlap threshold is met.
                valid_indices = []
                for step in (1, -1):
                    if step == 1:
                        context_index = max(choose_index)
                    else:
                        context_index = min(choose_index)
                    min_distance = frame_gap[0]
                    max_distance = frame_gap[1]
                    current_index = context_index + step * min_distance

                    while 0 <= current_index < v:
                        # Compute overlap.
                        overlap, overlap_a, overlap_b = calDelta(
                            ang_list[context_index],
                            ang_list[current_index],
                            dist_list[context_index],
                            dist_list[current_index],
                        )
                        # print("similarity",overlap_a,overlap_b)
                        min_overlap = overlap_range[0]
                        max_overlap = overlap_range[1]
                        if min_overlap <= overlap <= max_overlap:
                            valid_indices.append((current_index, overlap_a, overlap_b))
                        delta = np.abs(current_index - context_index)

                        # Stop once the camera has panned away too much.
                        if overlap < min_overlap or delta > max_distance:
                            break

                        current_index += step
                if valid_indices:
                    # Pick a random valid view. Index the resulting views.
                    num_options = len(valid_indices)
                    chosen = torch.randint(
                        0, num_options, size=tuple(), generator=generator
                    )
                    chosen, overlap_a, overlap_b = valid_indices[chosen]
                    choose_index.append(chosen)
                else:
                    break
            if len(choose_index) < num_context_views:
                continue
            context_left = min(choose_index)
            context_right = max(choose_index)
            delta = context_right - context_left
            choose_index.sort()
            # Pick non-repeated random target views.

            extrinsics = extrinsics[choose_index]
            intrinsics = intrinsics[choose_index]
            img_height, img_width = context_images.shape[-2:]
            img_width_new = target_size
            img_height_new = (
                round(img_height * (img_width_new / img_width) / _alignment)
                * _alignment
            )
            transform = T.Compose(
                [
                    T.Resize((img_height_new, img_width_new)),
                ]
            )
            imgs_list = transform(context_images)
            if img_height_new > target_size:
                start_y = (img_height_new - target_size) // 2
                context_images = context_images[
                    :, :, start_y : start_y + target_size, :
                ]
            imgs_list, intrinsics_list = resize_images_intrinsic(
                imgs_list,
                intrinsics,
                (img_height_new, img_width_new),
                (target_size, target_size),
            )
            pc_outputdir = os.path.join(outputdir, split, scene)
            im_outputdir = os.path.join(outputdir, "images", split, scene)
            plyGet(
                imgs_list[choose_index],
                scene,
                pc_outputdir,
                im_outputdir,
                device,
                dtype,
                vggt_model,
                depth_points="points",
                conf=conf,
                parse_depths=parse_depths,
            )
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--num_context_views",
        default=4,
        type=int,
        help="num_context_views",
    )
    parser.add_argument(
        "--frame_gap",
        default=[15, 135],
        type=int,
        help="frame_gap",
    )
    parser.add_argument(
        "--overlap_range",
        default=[5, 40],
        type=int,
        help="overlap_range",
    )
    parser.add_argument(
        "--conf",
        default=0.0,
        type=float,
        help="overlap_range",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="num_workers",
    )
    parser.add_argument(
        "--thread_id",
        default=0,
        type=int,
        help="thread_id",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device",
    )
    parser.add_argument(
        "--splits",
        default=["train", "test"],
        nargs="+",
        choices=["train", "test"],
        help="Splits need to process.",
    )
    parser.add_argument(
        "--parse_depths", action="store_true", help="Whether parse depths"
    )
    cfg = parser.parse_args()
    VGGT_model = VGGT().to(cfg.device)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL)
    VGGT_model.load_state_dict(state_dict)
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    chunk_paths = []
    for split in cfg.splits:
        root = Path(cfg.dataset_root) / split
        root_chunks = [entry for entry in root.iterdir() if ".json" not in str(entry)]
        root_chunks = sorted(root_chunks)
        chunk_paths.extend(root_chunks)
    chunk_paths_list = np.array_split(chunk_paths, cfg.num_workers)
    chunk_paths_ = chunk_paths_list[cfg.thread_id]

    for chunk_path in chunk_paths_:
        parse_scene(
            chunk_path,
            VGGT_model,
            cfg.num_context_views,
            cfg.output_root,
            cfg.device,
            cfg.frame_gap,
            cfg.overlap_range,
            dtype,
            cfg.conf,
            cfg.parse_depths,
        )
