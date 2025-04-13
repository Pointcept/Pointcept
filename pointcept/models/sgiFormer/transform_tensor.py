"""
# This file includes code adapted from SGIFormer:
# https://github.com/RayYoh/SGIFormer
# Original author: Lei Yao (rayyohhust@gmail.com)
"""

import random
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
from torch import Tensor
from pointcept.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class NormalizeColorT(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            assert isinstance(data_dict["color"], Tensor)
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict


@TRANSFORMS.register_module()
class Mask3DNormalizeColorT(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), scale=255):
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            assert isinstance(data_dict["color"], Tensor)
            scale = data_dict["color"].new_tensor(self.scale)
            mean = data_dict["color"].new_tensor(self.mean)
            std = data_dict["color"].new_tensor(self.std)
            data_dict["color"] = (data_dict["color"] / scale - mean) / std
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoordT(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = data_dict["coord"].mean(0)
            data_dict["coord"] -= centroid
            m = torch.max(torch.sqrt(torch.sum(data_dict["coord"] ** 2, dim=1)))
            data_dict["coord"] = data_dict["coord"] / m
        return data_dict


@TRANSFORMS.register_module()
class PositiveShiftT(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = data_dict["coord"].min(0)[0]
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class CenterShiftT(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            assert isinstance(data_dict["coord"], Tensor)
            coord = data_dict["coord"]
            x_min, y_min, z_min = coord.min(0)[0]
            x_max, y_max, _ = coord.max(0)[0]
            if self.apply_z:
                shift = coord.new_tensor(
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
                )
            else:
                shift = coord.new_tensor([(x_min + x_max) / 2, (y_min + y_max) / 2, 0])
            data_dict["coord"] -= shift
        return data_dict


@TRANSFORMS.register_module()
class PointClipT(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            assert isinstance(data_dict["coord"], Tensor)
            data_dict["coord"] = torch.clamp(
                data_dict["coord"],
                min=self.point_cloud_range[:3],
                max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropoutT(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            assert isinstance(data_dict["coord"], Tensor)
            n = len(data_dict["coord"])
            idx = torch.randperm(n)[: int(n * (1 - self.dropout_ratio))]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = torch.unique(torch.cat([idx, data_dict["sampled_index"]]))
                mask = torch.zeros_like(data_dict["segment"]).bool()
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = torch.where(mask[idx])[0]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
            if "superpoint" in data_dict.keys():
                data_dict["superpoint"] = torch.unique(
                    data_dict["superpoint"][idx], return_inverse=True
                )[1]
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateT(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict

        assert isinstance(data_dict["coord"], Tensor)
        coord = data_dict["coord"]

        angle = coord.new_tensor(
            [np.random.uniform(self.angle[0], self.angle[1]) * np.pi]
        )
        rot_cos, rot_sin = torch.cos(angle), torch.sin(angle)
        if self.axis == "x":
            rot_t = coord.new_tensor(
                [[1, 0, 0], [0, rot_cos, rot_sin], [0, -rot_sin, rot_cos]]
            )
        elif self.axis == "y":
            rot_t = coord.new_tensor(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]]
            )
        elif self.axis == "z":
            rot_t = coord.new_tensor(
                [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
            )
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(0)[0]
                x_max, y_max, z_max = data_dict["coord"].max(0)[0]
                center = coord.new_tensor(
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
                )
            else:
                center = coord.new_tensor(self.center)
            data_dict["coord"] -= center
            data_dict["coord"] = torch.mm(data_dict["coord"], rot_t)
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = torch.mm(data_dict["normal"], rot_t)
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngleT(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        assert isinstance(data_dict["coord"], Tensor)
        coord = data_dict["coord"]

        angle = coord.new_tensor([np.random.choice(self.angle) * np.pi])
        rot_cos, rot_sin = torch.cos(angle), torch.sin(angle)
        if self.axis == "x":
            rot_t = coord.new_tensor(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]]
            )
        elif self.axis == "y":
            rot_t = coord.new_tensor(
                [[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]]
            )
        elif self.axis == "z":
            rot_t = coord.new_tensor(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
            )
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(0)[0]
                x_max, y_max, z_max = data_dict["coord"].max(0)[0]
                center = coord.new_tensor(
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
                )
            else:
                center = coord.new_tensor(self.center)
            data_dict["coord"] -= center
            data_dict["coord"] = torch.mm(data_dict["coord"], rot_t)
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = torch.mm(data_dict["normal"], rot_t)
        return data_dict


@TRANSFORMS.register_module()
class RandomScaleT(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            assert isinstance(data_dict["coord"], Tensor)
            device = data_dict["coord"].device
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            scale = torch.tensor(scale).to(device)
            data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlipT(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class Mask3DRandomFlipT(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        for i in (0, 1):
            if np.random.rand() < self.p:
                if "coord" in data_dict.keys():
                    coord_max = data_dict["coord"].max(0)[0]
                    data_dict["coord"][:, i] = coord_max[i] - data_dict["coord"][:, i]
                if "normal" in data_dict.keys():
                    data_dict["normal"][:, i] = -data_dict["normal"][:, i]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitterT(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            assert isinstance(data_dict["coord"], Tensor)
            device = data_dict["coord"].device
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            jitter = torch.tensor(jitter).to(device)
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitterT(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            assert isinstance(data_dict["coord"], Tensor)
            device = data_dict["coord"].device
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            jitter = torch.tensor(jitter).to(device)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrastT(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            assert isinstance(data_dict["color"], Tensor)
            device = data_dict["color"].device
            lo = data_dict["color"].min(0, keepdim=True)[0]
            hi = data_dict["color"].max(0, keepdim=True)[0]
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            blend_factor = torch.tensor([blend_factor]).to(device)
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslationT(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            assert isinstance(data_dict["color"], Tensor)
            device = data_dict["color"].device
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            tr = torch.tensor(tr).to(device)
            data_dict["color"][:, :3] = torch.clamp(
                tr + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitterT(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            assert isinstance(data_dict["color"], Tensor)
            device = data_dict["color"].device
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise = torch.tensor(noise).to(device)
            noise *= self.std * 255
            data_dict["color"][:, :3] = torch.clamp(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorGrayScaleT(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        assert isinstance(color, Tensor)
        device = color.device
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)
        gray = torch.tensor(gray).to(device)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDropT(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            assert isinstance(data_dict["color"], Tensor)
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


@TRANSFORMS.register_module()
class ElasticDistortionT(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            assert isinstance(data_dict["coord"], Tensor)
            device = data_dict["coord"].device
            coord = data_dict["coord"].numpy()
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    coord = self.elastic_distortion(coord, granularity, magnitude)
            data_dict["coord"] = torch.tensor(coord).to(device)
        return data_dict


@TRANSFORMS.register_module()
class GridSampleT(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        assert isinstance(data_dict["coord"], Tensor)
        coord = data_dict["coord"]
        device = coord.device

        scaled_coord = data_dict["coord"] / coord.new_tensor(self.grid_size)
        grid_coord = torch.floor(scaled_coord).long()
        min_coord = grid_coord.min(0)[0]
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * coord.new_tensor(self.grid_size)

        key = self.hash(grid_coord.numpy())
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        inverse = torch.tensor(inverse, dtype=torch.long, device=device)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            idx_unique = torch.tensor(idx_unique, dtype=torch.long, device=device)
            idx_sort = torch.tensor(idx_sort, dtype=torch.long, device=device)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = torch.unique(
                    torch.cat([idx_unique, data_dict["sampled_index"]])
                )
                mask = torch.zeros_like(data_dict["segment"]).bool()
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = torch.where(mask[idx_unique])[0]

            if self.return_inverse:
                data_dict["inverse"] = torch.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.view(1, 3)
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = torch.sum(
                        displacement * data_dict["normal"], dim=-1, keepdim=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                idx_part = torch.tensor(idx_part, dtype=torch.long, device=device)
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = torch.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.view(1, 3)
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = torch.sum(
                            displacement * data_dict["normal"], dim=-1, keepdim=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class SphereCropT(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        assert isinstance(data_dict["coord"], Tensor)
        device = data_dict["coord"].device
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = torch.arange(
                    data_dict["coord"].shape[0], device=device
                )
            data_part_list = []
            # coord_list, color_list, dist2_list, idx_list, offset_list = [], [], [], [], []
            if data_dict["coord"].shape[0] > point_max:
                coord_p, idx_uni = torch.rand(
                    data_dict["coord"].shape[0], device=device
                ) * 1e-3, torch.tensor([], device=device)
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = torch.argmin(coord_p)
                    dist2 = torch.sum(
                        torch.pow(data_dict["coord"] - data_dict["coord"][init_idx], 2),
                        1,
                    )
                    idx_crop = torch.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "grid_coord" in data_dict.keys():
                        data_crop_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    if "displacement" in data_dict.keys():
                        data_crop_dict["displacement"] = data_dict["displacement"][
                            idx_crop
                        ]
                    if "strength" in data_dict.keys():
                        data_crop_dict["strength"] = data_dict["strength"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = torch.square(
                        1
                        - data_crop_dict["weight"] / torch.max(data_crop_dict["weight"])
                    )
                    coord_p[idx_crop] += delta
                    idx_uni = torch.unique(
                        torch.cat((idx_uni, data_crop_dict["index"]))
                    )
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = torch.zeros(data_dict["coord"].shape[0]).to(
                    device
                )
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    torch.randint(data_dict["coord"].shape[0], (1,), device=device)
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = torch.argsort(
                torch.sum(torch.square(data_dict["coord"] - center), 1)
            )[:point_max]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "displacement" in data_dict.keys():
                data_dict["displacement"] = data_dict["displacement"][idx_crop]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx_crop]
            if "superpoint" in data_dict.keys():
                data_dict["superpoint"] = torch.unique(
                    data_dict["superpoint"][idx_crop], return_inverse=True
                )[1]
            if "elastic_coord" in data_dict.keys():
                data_dict["elastic_coord"] = data_dict["elastic_coord"][idx_crop]
        return data_dict


@TRANSFORMS.register_module()
class S3DISValSphereCropT(object):
    def __init__(self, point_max=3000000, sample_rate=0.9, mode="center"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        self.mode = mode

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        assert isinstance(data_dict["coord"], Tensor)
        device = data_dict["coord"].device

        if data_dict["coord"].shape[0] > self.point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    torch.randint(data_dict["coord"].shape[0], (1,), device=device)
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            point_max = int(self.sample_rate * data_dict["coord"].shape[0])
            idx_crop = torch.argsort(
                torch.sum(torch.square(data_dict["coord"] - center), 1)
            )[:point_max]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "displacement" in data_dict.keys():
                data_dict["displacement"] = data_dict["displacement"][idx_crop]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx_crop]
            if "superpoint" in data_dict.keys():
                data_dict["superpoint"] = torch.unique(
                    data_dict["superpoint"][idx_crop], return_inverse=True
                )[1]
            if "elastic_coord" in data_dict.keys():
                data_dict["elastic_coord"] = data_dict["elastic_coord"][idx_crop]
        return data_dict


@TRANSFORMS.register_module()
class ShufflePointT(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        assert isinstance(data_dict["coord"], Tensor)
        device = data_dict["coord"].device
        shuffle_index = torch.randperm(data_dict["coord"].shape[0], device=device)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][shuffle_index]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][shuffle_index]
        if "displacement" in data_dict.keys():
            data_dict["displacement"] = data_dict["displacement"][shuffle_index]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][shuffle_index]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][shuffle_index]
        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"][shuffle_index]
        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"][shuffle_index]
        return data_dict


@TRANSFORMS.register_module()
class InstanceParserT(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        assert isinstance(coord, Tensor)
        device = coord.device

        mask = ~torch.isin(segment, coord.new_tensor(self.segment_ignore_index))
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = torch.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = (
            torch.ones((coord.shape[0], 3), device=device, dtype=coord.dtype)
            * self.instance_ignore_index
        )
        bbox = (
            torch.ones((instance_num, 8), device=device, dtype=coord.dtype)
            * self.instance_ignore_index
        )
        vacancy = [index for index in self.segment_ignore_index if index >= 0]
        vacancy = torch.tensor(vacancy, dtype=coord.dtype, device=device)

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)[0]
            bbox_max = coord_.max(0)[0]
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = torch.zeros(1, dtype=coord_.dtype, device=device)
            bbox_class = torch.tensor(
                [segment[mask_][0]], dtype=coord_.dtype, device=device
            )
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= torch.sum(bbox_class > vacancy)
            centroid[mask_] = bbox_centroid
            bbox[instance_id] = torch.cat(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


@TRANSFORMS.register_module()
class MeanShiftT(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            assert isinstance(data_dict["coord"], Tensor)
            coord = data_dict["coord"]
            coord -= coord.mean(0)
            data_dict["coord"] = coord
        return data_dict


@TRANSFORMS.register_module()
class Mask3DShiftT(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            assert isinstance(data_dict["coord"], Tensor)
            coord = data_dict["coord"]
            coord -= coord.mean(0)
            coord += torch.rand_like(coord) * (coord.max(0)[0] - coord.min(0)[0]) / 2
            data_dict["coord"] = coord
        return data_dict


@TRANSFORMS.register_module()
class CustomGridSampleT(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "elastic_coord" in data_dict.keys()
        device = data_dict["elastic_coord"].device

        scaled_coord = data_dict["elastic_coord"] - data_dict["elastic_coord"].min(0)[0]
        grid_coord = torch.floor(scaled_coord).long()
        min_coord = grid_coord.min(0)[0]
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * scaled_coord.new_tensor(self.grid_size)

        key = self.hash(grid_coord.numpy())
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            idx_unique = torch.tensor(idx_unique, dtype=torch.long, device=device)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = torch.unique(
                    torch.cat([idx_unique, data_dict["sampled_index"]])
                )
                mask = torch.zeros_like(data_dict["segment"]).bool()
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = torch.where(mask[idx_unique])[0]

            if self.return_inverse:
                inverse = torch.tensor(inverse, dtype=torch.long, device=device)
                idx_sort = torch.tensor(idx_sort, dtype=torch.long, device=device)
                data_dict["inverse"] = torch.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.view(1, 3)
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = torch.sum(
                        displacement * data_dict["normal"], dim=-1, keepdim=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                idx_part = torch.tensor(idx_part, dtype=torch.long, device=device)
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = torch.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.view(1, 3)
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = torch.sum(
                            displacement * data_dict["normal"], dim=-1, keepdim=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class CustomGridSampleT_(CustomGridSampleT):
    def __call__(self, data_dict):
        assert "elastic_coord" in data_dict.keys()
        device = data_dict["elastic_coord"].device

        # - data_dict["elastic_coord"].min(0)[0]
        scaled_coord = data_dict["elastic_coord"]
        grid_coord = torch.floor(scaled_coord).long()
        min_coord = grid_coord.min(0)[0]
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * scaled_coord.new_tensor(self.grid_size)

        key = self.hash(grid_coord.numpy())
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            idx_unique = torch.tensor(idx_unique, dtype=torch.long, device=device)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = torch.unique(
                    torch.cat([idx_unique, data_dict["sampled_index"]])
                )
                mask = torch.zeros_like(data_dict["segment"]).bool()
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = torch.where(mask[idx_unique])[0]

            if self.return_inverse:
                inverse = torch.tensor(inverse, dtype=torch.long, device=device)
                idx_sort = torch.tensor(idx_sort, dtype=torch.long, device=device)
                data_dict["inverse"] = torch.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.view(1, 3)
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = torch.sum(
                        displacement * data_dict["normal"], dim=-1, keepdim=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                idx_part = torch.tensor(idx_part, dtype=torch.long, device=device)
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = torch.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.view(1, 3)
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = torch.sum(
                            displacement * data_dict["normal"], dim=-1, keepdim=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError


@TRANSFORMS.register_module()
class CustomElasticDistortionT(object):
    def __init__(self, grid_size=0.02, distortion_params=None, p=0.5):
        self.grid_size = grid_size
        self.distortion_params = (
            [[6, 40], [20, 160]] if distortion_params is None else distortion_params
        )
        self.p = p

    @staticmethod
    def elastic(x, gran, mag):
        """Private function for elastic transform to a points.

        Args:
            x (ndarray): Point cloud.
            gran (List[float]): Size of the noise grid (in same scale[m/cm]
                as the voxel grid).
            mag: (List[float]): Noise multiplier.

        Returns:
            dict: Results after elastic, 'points' is updated
                in the result dict.
        """
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        noise_dim = (np.abs(x).max(0) // gran).astype(int) + 3
        noise = [
            np.random.randn(noise_dim[0], noise_dim[1], noise_dim[2]).astype("float32")
            for _ in range(3)
        ]

        for blur in [blur0, blur1, blur2, blur0, blur1, blur2]:
            noise = [
                scipy.ndimage.filters.convolve(n, blur, mode="constant", cval=0)
                for n in noise
            ]

        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in noise_dim]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0
            )
            for n in noise
        ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * mag

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            assert isinstance(data_dict["coord"], Tensor)
            device = data_dict["coord"].device
            data_dict["elastic_coord"] = data_dict["coord"].numpy() / self.grid_size
            if random.random() < self.p:
                for granularity, magnitude in self.distortion_params:
                    data_dict["elastic_coord"] = self.elastic(
                        data_dict["elastic_coord"], granularity, magnitude
                    )
            data_dict["elastic_coord"] = torch.tensor(data_dict["elastic_coord"]).to(
                device
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomTranslationT(object):
    def __init__(self, shift=(0.1, 0.1, 0.1)):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            assert isinstance(data_dict["coord"], Tensor)
            coord = data_dict["coord"]
            if self.shift is not None:
                translation_std = np.array(self.shift, dtype=np.float32)
                trans_factor = np.random.normal(scale=translation_std, size=3).T
                trans_factor = coord.new_tensor(trans_factor)
                data_dict["coord"] += trans_factor
        return data_dict


@TRANSFORMS.register_module()
class RandomShiftT(object):
    def __init__(self, shift=(0.1, 0.1, 0.1)):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            assert isinstance(data_dict["coord"], Tensor)
            coord = data_dict["coord"]
            if self.shift is not None:
                shift_std = np.array(self.shift, dtype=np.float32)
                shift_factor = np.random.normal(scale=shift_std, size=3).T
                shift_factor = coord.new_tensor(shift_factor)
                data_dict["coord"] += shift_factor
        return data_dict


@TRANSFORMS.register_module()
class SwapChairAndFloorT(object):
    def __call__(self, data_dict):
        if "segment" in data_dict.keys():
            assert isinstance(data_dict["segment"], Tensor)
            mask = data_dict["segment"].clone()
            mask[data_dict["segment"] == 1] = 2
            mask[data_dict["segment"] == 2] = 1
            data_dict["segment"] = mask
        return data_dict


@TRANSFORMS.register_module()
class InsClassMapT(object):
    def __init__(self, ins_cls_ids):
        self.ins_cls_ids = ins_cls_ids
        self.id_to_index = {cls_id: i for i, cls_id in enumerate(self.ins_cls_ids)}
        self.index_to_id = {i: cls_id for i, cls_id in enumerate(self.ins_cls_ids)}

    def __call__(self, data_dict):
        assert "segment" in data_dict.keys()
        assert isinstance(data_dict["segment"], Tensor)
        new_segment = torch.ones_like(data_dict["segment"]) * -1
        for i, ins_cls_id in enumerate(self.ins_cls_ids):
            new_segment[data_dict["segment"] == ins_cls_id] = i
        data_dict["segment"] = new_segment
        return data_dict

    def reverse_map(self, predicted_classes: Tensor) -> Tensor:
        """
        Args:
            predicted_classes (Tensor): Tensor of predicted class IDs in the range [0-83].

        Returns:
            Tensor: Tensor of class IDs mapped back to the original [0-99] range.
        """
        vectorized_map = np.vectorize(lambda x: self.index_to_id.get(x, -1))
        return vectorized_map(predicted_classes)
