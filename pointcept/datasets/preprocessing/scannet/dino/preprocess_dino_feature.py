import os
import argparse
import einops
import torch
import torch.nn.functional as F
import torchvision
import tqdm
import cv2
import camtools as ct
import open3d as o3d
import zlib
import imageio
import struct
import numpy as np
import torch_scatter
from pathlib import Path


class RGBDFrame:
    def __init__(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.v2.imread(self.color_data)


class SensorData:
    COMPRESSION_TYPE_COLOR = {
        -1: "unknown",
        0: "raw",
        1: "png",
        2: "jpeg",
    }
    COMPRESSION_TYPE_DEPTH = {
        -1: "unknown",
        0: "raw_ushort",
        1: "zlib_ushort",
        2: "occi_ushort",
    }

    def __init__(self, filename):
        self.version = 4
        f = open(filename, "rb")
        version = struct.unpack("I", f.read(4))[0]
        assert self.version == version
        strlen = struct.unpack("Q", f.read(8))[0]
        self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
        self.intrinsic_color = np.asarray(
            struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.extrinsic_color = np.asarray(
            struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.intrinsic_depth = np.asarray(
            struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.extrinsic_depth = np.asarray(
            struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.color_compression_type = self.COMPRESSION_TYPE_COLOR[
            struct.unpack("i", f.read(4))[0]
        ]
        self.depth_compression_type = self.COMPRESSION_TYPE_DEPTH[
            struct.unpack("i", f.read(4))[0]
        ]
        self.color_width = struct.unpack("I", f.read(4))[0]
        self.color_height = struct.unpack("I", f.read(4))[0]
        self.depth_width = struct.unpack("I", f.read(4))[0]
        self.depth_height = struct.unpack("I", f.read(4))[0]
        self.depth_shift = struct.unpack("f", f.read(4))[0]
        self.num_frames = struct.unpack("Q", f.read(8))[0]
        self.file_handle = f

    def export(
        self,
        frame_skip=20,
        export_color=True,
        export_depth=True,
        export_pose=True,
    ):
        for i in range(self.num_frames):
            if i % frame_skip != 0:
                self.file_handle.seek(16 * 4 + 8 + 8, 1)  # skip pose, timestamp
                color_size_bytes = struct.unpack("Q", self.file_handle.read(8))[0]
                depth_size_bytes = struct.unpack("Q", self.file_handle.read(8))[0]
                self.file_handle.seek(color_size_bytes + depth_size_bytes, 1)
                continue
            else:
                frame = RGBDFrame(self.file_handle)
                data_dict = {}
                if export_color:
                    color = frame.decompress_color(self.color_compression_type)
                    data_dict["color"] = color
                if export_depth:
                    depth = frame.decompress_depth(self.depth_compression_type)
                    depth = np.frombuffer(depth, dtype=np.uint16).reshape(
                        self.depth_height, self.depth_width
                    )
                    data_dict["depth"] = depth
                if export_pose:
                    pose = frame.camera_to_world
                    data_dict["pose"] = pose
                yield data_dict

    def __del__(self):
        self.file_handle.close()


def ray_distance_to_z_depth(ray_depth, K):
    height, width = ray_depth.shape

    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u_norm = (u_grid - cx) / fx
    v_norm = (v_grid - cy) / fy

    norm_square = u_norm**2 + v_norm**2

    z_depth = ray_depth / np.sqrt(norm_square + 1)
    return z_depth


def center_crop(image, crop_ratio=1.0, patch_size=None):
    if len(image.shape) == 2:
        height, width = image.shape
    elif len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        raise ValueError("Invalid image shape")
    if patch_size is not None:
        crop_h = int(height * crop_ratio // patch_size * patch_size)
        crop_w = int(width * crop_ratio // patch_size * patch_size)
    else:
        crop_h = int(height * crop_ratio)
        crop_w = int(width * crop_ratio)

    # Calculate the cropping box
    start_h = (height - crop_h) // 2
    start_w = (width - crop_w) // 2

    # Perform the center crop
    cropped_image = image[start_h : start_h + crop_h, start_w : start_w + crop_w]

    return cropped_image


def parsing_scene(
    scene_path,
    output_root,
    split,
    model,
    frame_skip=20,
    grid_size=0.08,
    crop_ratio=0.95,
    device="cuda",
):
    print(f"Parsing scene: {scene_path.name}")
    device = torch.device(device)
    scene_path = Path(scene_path)
    sensor_reader = SensorData(scene_path / f"{scene_path.name}.sens")
    mesh = o3d.io.read_triangle_mesh(
        str(scene_path / f"{scene_path.name}_vh_clean_2.ply")
    )
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    scene_coord = []
    scene_feat = []
    scene_count = []
    for data in tqdm.tqdm(
        sensor_reader.export(frame_skip=frame_skip),
        total=sensor_reader.num_frames // frame_skip,
    ):
        height, width = data["depth"].shape
        K = sensor_reader.intrinsic_depth[:3, :3]
        T = data["pose"]
        if np.isnan(T).any() or np.isinf(T).any():
            continue
        depth = ct.raycast.mesh_to_depth(
            mesh=mesh, K=K, T=np.linalg.inv(T), height=height, width=width
        )
        depth = ray_distance_to_z_depth(depth, K)
        depth = center_crop(depth, crop_ratio, model.patch_size)
        height_, width_ = depth.shape
        pixel = np.transpose(np.indices((width_, height_)), (2, 1, 0))
        pixel = pixel.reshape((-1, 2))
        pixel = np.hstack((pixel, np.ones((pixel.shape[0], 1))))
        depth = depth.reshape((-1, 1))
        valid = ~np.isinf(depth).squeeze(-1)
        coord = depth[valid] * (np.linalg.inv(K) @ pixel[valid].T).T  # coord_camera
        coord = coord @ T[:3, :3].T + T[:3, 3]

        color = cv2.resize(
            data["color"], (width, height), interpolation=cv2.INTER_LINEAR
        )
        color = center_crop(color, crop_ratio, model.patch_size)
        with torch.inference_mode():
            color_t = transform(color).unsqueeze(0).to(device)
            feat_t = model.forward_features(color_t)["x_norm_patchtokens"]
            feat_t = einops.rearrange(
                feat_t, "1 (h w) c -> 1 c h w", w=width_ // model.patch_size
            )
            feat_t = F.interpolate(feat_t, (height_, width_), mode="bilinear")
            feat_t = einops.rearrange(feat_t, "1 c h w -> (h w) c")[valid]
            coord_t = torch.tensor(coord, dtype=torch.float32).to(device)
            scene_coord.append(coord_t)
            scene_feat.append(feat_t)
            scene_count.append(
                torch.ones(coord_t.shape[0], dtype=torch.long, device=device)
            )
            scene_coord = torch.concatenate(scene_coord, dim=0)
            scene_feat = torch.concatenate(scene_feat, dim=0)
            scene_count = torch.concatenate(scene_count, dim=0)

            # grid sampling
            grid_coord = torch.floor_divide(scene_coord, grid_size).to(torch.int32)
            grid_coord, cluster = torch.unique(
                grid_coord, sorted=True, return_inverse=True, dim=0
            )
            scene_coord = [
                torch_scatter.scatter(scene_coord, cluster, reduce="mean", dim=0)
            ]
            scene_feat = [
                torch_scatter.scatter(scene_feat, cluster, reduce="sum", dim=0)
            ]
            scene_count = [
                torch_scatter.scatter(scene_count, cluster, reduce="sum", dim=0)
            ]

        # color = color.reshape((-1, 3))[valid]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(coord)
        # pcd.colors = o3d.utility.Vector3dVector(color / 255)
        # o3d.visualization.draw_geometries([pcd])

    scene_coord = scene_coord[0]
    scene_feat = scene_feat[0] / scene_count[0].unsqueeze(-1)

    scene_coord = scene_coord.half().cpu().numpy()
    scene_feat = scene_feat.half().cpu().numpy()
    np.savez(
        Path(output_root) / split / f"{scene_path.name}.npz",
        coord=scene_coord,
        feat=scene_feat,
    )


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
        "--scene_list",
        required=True,
        help="Path to scene list need to process",
    )
    parser.add_argument(
        "--frame_skip",
        default=10,
        help="Frame skip for processing",
    )
    parser.add_argument(
        "--grid_size",
        default=0.08,
        help="Grid size for sampling",
    )
    parser.add_argument(
        "--crop_ratio",
        default=0.95,
        help="Crop ratio for center crop",
    )

    args = parser.parse_args()
    scene_list = np.loadtxt(args.scene_list, dtype=str)
    if "train" in args.scene_list:
        split = "train"
        folder = "scans"
    elif "val" in args.scene_list:
        split = "val"
        folder = "scans"
    else:
        split = "test"
        folder = "scans_test"

    os.makedirs(Path(args.output_root) / split, exist_ok=True)

    device = torch.device("cuda")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14").to(device)
    model.eval()
    for scene in scene_list:
        parsing_scene(
            scene_path=Path(args.dataset_root) / folder / scene,
            output_root=args.output_root,
            split=split,
            frame_skip=args.frame_skip,
            grid_size=args.grid_size,
            crop_ratio=args.crop_ratio,
            model=model,
            device="cuda",
        )

    # parsing_scene(
    #     scene_path=Path("/mnt/e/datasets/raw/scannet/scans/scene0230_00"),
    #     output_root=args.output_root,
    #     split=split,
    #     frame_skip=args.frame_skip,
    #     grid_size=args.grid_size,
    #     crop_ratio=args.crop_ratio,
    #     model=model,
    #     device="cuda",
    # )
