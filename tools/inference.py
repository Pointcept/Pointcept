import time
import os
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement # plyfile for saving final PLY
import torch
from collections import OrderedDict
import sys

# Ensure Pointcept modules are discoverable (important when running directly)
# This assumes you run the script from the root of your Pointcept repository
# e.g., cd /home/aiml/HDD/amit/Pointcept/ && python tools/inference.py ...
if os.path.abspath(os.path.curdir) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.curdir))


from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import collate_fn

# Define constants (should match your training config)
IGNORE_INDEX = -1 # Make sure this matches your config
# Get class names from your config (these should be in the exact order of your mapped IDs)
# For example, if you mapped raw 1->0, 2->1, 8->2, 9->3, 40->4
# YOUR_CLASS_NAMES = ["wall", "floor", "door", "window", "roof"]
# We'll load them from the config for consistency.

CLASS_LABELS_FULL = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_FULL = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

YOUR_RAW_LABEL_VALUES = [1, 2, 8, 9, 40] # These are your actual raw IDs used during training
YOUR_CLASS_NAMES = ["wall", "floor", "door", "window", "roof"]
ID_TO_DISPLAY_COLOR = np.zeros((len(YOUR_CLASS_NAMES) + 1, 3), dtype=np.uint8)
for mapped_id, raw_id in enumerate(YOUR_RAW_LABEL_VALUES):
    if raw_id in SCANNET_COLOR_MAP_FULL:
        ID_TO_DISPLAY_COLOR[mapped_id] = SCANNET_COLOR_MAP_FULL[raw_id]
    else:
        # Fallback for raw_ids not found in SCANNET_COLOR_MAP_FULL (e.g., if you added a new one)
        # Assign a distinct color if needed, or a generic grey
        print(f"Warning: Raw ID {raw_id} for mapped ID {mapped_id} not found in SCANNET_COLOR_MAP_FULL. Assigning default color.")
        ID_TO_DISPLAY_COLOR[mapped_id] = [100, 100, 100] # Default grey

# Assign color for IGNORE_INDEX (mapped to the last index: len(YOUR_CLASS_NAMES))
ID_TO_DISPLAY_COLOR[len(YOUR_CLASS_NAMES)] = [0, 0, 0] # Black for ignored points

def read_ply_simple(filepath):
    """Reads a PLY file containing x,y,z,r,g,b, (optional normals)"""
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        vertices = plydata["vertex"].data
        coords = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
        colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.uint8)

        normals = None
        if "nx" in vertices.dtype.names and "ny" in vertices.dtype.names and "nz" in vertices.dtype.names:
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T.astype(np.float32)
        
        return coords, colors, normals
    return None, None, None

def save_ply_with_labels(coords, colors, labels, output_filepath, class_names):
    """Saves a PLY file with predicted labels."""
    # Ensure labels are within the valid range for uchar (0-255) if saving as uchar
    # If more classes, use int
    num_classes = len(class_names)
    label_dtype = "u1" if num_classes <= 256 else "i4"

    # Define the PLY vertex elements
    vertex = np.empty(coords.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', label_dtype) # predicted label
    ])

    vertex['x'] = coords[:, 0]
    vertex['y'] = coords[:, 1]
    vertex['z'] = coords[:, 2]

    # Create a temporary array for visual mapping, handling IGNORE_INDEX
    visual_labels_for_color = np.copy(labels)
    # Map IGNORE_INDEX (-1) to the last index in our ID_TO_DISPLAY_COLOR array
    # The last index is len(YOUR_CLASS_NAMES)
    visual_labels_for_color[visual_labels_for_color == IGNORE_INDEX] = len(YOUR_CLASS_NAMES) 
    
    # Ensure all indices are within the valid range [0, len(YOUR_CLASS_NAMES)]
    clamped_visual_labels = np.clip(visual_labels_for_color, 0, len(YOUR_CLASS_NAMES))
    
    # Look up the color using the mapped/clamped labels
    predicted_colors = ID_TO_DISPLAY_COLOR[clamped_visual_labels]

    vertex['red'] = predicted_colors[:, 0]
    vertex['green'] = predicted_colors[:, 1]
    vertex['blue'] = predicted_colors[:, 2]
    vertex['label'] = labels # ensure labels are integer type

    ply_element = PlyElement.describe(vertex, 'vertex')
    PlyData([ply_element]).write(output_filepath)
    print(f"Saved prediction to {output_filepath}")


def inference_single_file(args):
    # 1. Load config and build model
    cfg = Config.fromfile(args.config_file)
    model = build_model(cfg.model)

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{args.checkpoint_path}'")
    
    # Load the checkpoint dictionary
    # Use map_location to ensure it loads to CPU first, then move to GPU later
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

    # Extract state_dict and process keys (handle 'module.' prefix)
    # This part mimics TesterBase's build_model, assuming single GPU inference
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("module."):
            key = key[7:] # Remove 'module.' prefix for single-GPU model
        weight[key] = value
        
    # # for cpu inference bias is not required
    # keys_to_remove = []
    # for key in weight.keys():
    #     # Find all keys that end with '.bias' and are part of the 'cpe' module
    #     if "cpe" in key and key.endswith(".bias"):
    #         keys_to_remove.append(key)

    # print(f"Removing {len(keys_to_remove)} bias keys from checkpoint for CPU inference...")
    # for key in keys_to_remove:
    #     del weight[key]

    model.load_state_dict(weight, strict=False)
    #model.load_state_dict(weight, strict=True)
    print("Checkpoint loaded successfully.")

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Get class names from config (important for saving output)
    class_names = cfg.data.names
    num_classes = cfg.data.num_classes
    
    # 2. Load the input PLY file
    print(f"Loading input file: {args.input_file}")
    coords_orig, colors_orig, normals_orig = read_ply_simple(args.input_file)

    if coords_orig is None or len(coords_orig) == 0:
        print(f"Error: Could not load or found no points in {args.input_file}")
        return

    # Use Open3D to estimate normals if not provided in PLY (and if model expects them)
    # The config expects normals, so we should provide them.
    if normals_orig is None:
        print("Normals not found in PLY, estimating with Open3D...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_orig)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)
        normals_orig = np.array(pcd.normals, dtype=np.float32)
    
    if args.initial_downsample_voxel_size is not None:
        print(f"Original points: {len(coords_orig)}")
        
        # Create Open3D point cloud
        pcd_initial = o3d.geometry.PointCloud()
        pcd_initial.points = o3d.utility.Vector3dVector(coords_orig)
        pcd_initial.colors = o3d.utility.Vector3dVector(colors_orig / 255.0) # Open3D expects [0,1]
        if normals_orig is not None:
            pcd_initial.normals = o3d.utility.Vector3dVector(normals_orig)

        print(f"Applying initial Open3D voxel downsampling with size: {args.initial_downsample_voxel_size}")
        pcd_downsampled = pcd_initial.voxel_down_sample(voxel_size=args.initial_downsample_voxel_size)
        
        # Update the original data_dict_full_scene to be the downsampled version
        coords_orig = np.asarray(pcd_downsampled.points, dtype=np.float32)
        colors_orig = (np.asarray(pcd_downsampled.colors) * 255).astype(np.uint8)
        
        # Handle normals after downsampling (they might be lost or need re-estimation)
        if pcd_downsampled.has_normals():
            normals_orig = np.asarray(pcd_downsampled.normals, dtype=np.float32)
        else:
            # If normals are critical for your model, you might re-estimate them here
            # using pcd_downsampled.estimate_normals() if they were lost.
            # For now, we'll just set to None if they were lost and not re-estimated.
            print("Warning: Normals were not preserved after initial downsampling. Setting to None.")
            normals_orig = None 
            
        print(f"Points after initial downsampling: {len(coords_orig)}")
    
    # Create an initial data_dict that mimics the dataset output before transforms
    data_dict_full_scene = {
        "coord": coords_orig,
        "color": colors_orig,
        "normal": normals_orig,
        "name": os.path.basename(args.input_file).replace(".ply", "")
        # 'segment' and 'instance' are not needed for inference
    }

    # 3. Apply initial transforms (cfg.data.test.transform)
    # These transforms operate on the whole point cloud before fragmentation
    initial_transforms_raw = cfg.data.test.transform 
    if isinstance(initial_transforms_raw, Compose):
        initial_transforms = initial_transforms_raw
    else:
        # If it's a list of dicts (standard), then wrap it
        initial_transforms = Compose(initial_transforms_raw)
    processed_full_scene_dict = initial_transforms(data_dict_full_scene)

    # 4. Create the voxelization/fragmentation transform (cfg.data.test.test_cfg.voxelize)
    # This transform generates the 'fragment_list'
    # We must explicitly add return_inverse=True here as well, as it's needed later
    voxelize_cfg_dict = cfg.data.test.test_cfg.voxelize.copy()
    voxelize_cfg_dict['return_inverse'] = True # Ensure inverse is returned for mapping back
    voxelize_transform = Compose([voxelize_cfg_dict])
    
    # The voxelize_transform will take a single data_dict and return a list of fragments.
    # It wraps the GridSample logic used in test_mode.
    # The output from this step is equivalent to the 'fragment_list' in test.py
    fragment_list = voxelize_transform(processed_full_scene_dict) 
    
    # 5. Create the post-voxelization transforms (cfg.data.test.test_cfg.post_transform)
    # These transforms operate on each individual fragment
    post_voxelize_transforms_raw = cfg.data.test.test_cfg.post_transform
    if isinstance(post_voxelize_transforms_raw, Compose):
        post_voxelize_transforms = post_voxelize_transforms_raw
    else:
        post_voxelize_transforms = Compose(post_voxelize_transforms_raw)

     # Prepare for accumulating raw logits for precise reconstruction
    # Initialize a full_logits array for the original point cloud
    full_logits = np.zeros((len(coords_orig), num_classes), dtype=np.float32)
    
    # 6. Run inference for each fragment
    print(f"Processing {len(fragment_list)} fragments...")
    with torch.no_grad(): # Disable gradient calculation for inference
        for i, fragment_data in enumerate(fragment_list):
            # Apply post-voxelization transforms to the current fragment
            # This prepares the fragment into the format expected by the model's forward pass
            processed_fragment = post_voxelize_transforms(fragment_data)

            # Collate the single fragment into a "batch" format expected by the model
            # Even for a single fragment, collate_fn expects a list of items
            input_dict = collate_fn([processed_fragment])
            
            # Move data to GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    if args.device == 'cuda':
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    else:
                        input_dict[key] = input_dict[key]

            outputs = model(input_dict)
            
            # Extract raw logits (before softmax/argmax)
            logits_fragment = outputs["seg_logits"] if isinstance(outputs, dict) else outputs
            
            # Move logits and inverse indices to CPU and convert to numpy
            logits_fragment_np = logits_fragment.cpu().numpy()
            # inverse_indices_np = input_dict["inverse"].cpu().numpy()
            inverse_indices_np = input_dict["index"].cpu().numpy()

            # Accumulate logits onto the full_logits array using the inverse indices
            # This sums contributions from overlapping fragments/voxel regions
            full_logits[inverse_indices_np] += logits_fragment_np
            
            print(f"Processed fragment {i+1}/{len(fragment_list)} for scene {processed_full_scene_dict['name']}")

    # 7. Reconstruct final prediction on original point cloud
    # Apply argmax to the accumulated logits to get the final predicted class ID for each point
    full_predictions = np.argmax(full_logits, axis=1).astype(np.int32)
    
    # 8. Save the results as a new PLY file
    output_basename = processed_full_scene_dict["name"] + "_pred.ply"
    output_filepath = os.path.join(args.output_dir, output_basename)
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_ply_with_labels(coords_orig, colors_orig, full_predictions, output_filepath, class_names)

    print("Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Pointcept model on a single PLY file.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for inference ('cuda' or 'cpu').")
    parser.add_argument("--config_file", required=True, help="Path to your model configuration file (e.g., configs/my_dataset/semseg-pt-v3m1-0-nwd.py).")
    parser.add_argument("--checkpoint_path", required=True, help="Path to your trained model checkpoint (.pth file).")
    parser.add_argument("--input_file", required=True, help="Path to the input PLY file for inference.")
    parser.add_argument("--output_dir", default="inference_results", help="Directory to save the predicted PLY file.")
    parser.add_argument(
        "--initial_downsample_voxel_size",
        type=float,
        default=None, # By default, no initial downsampling (set to None)
        help="If specified, applies initial Open3D voxel downsampling. Value is the voxel size. "
             "E.g., 0.05. If None, no initial downsampling is performed."
    )
    args = parser.parse_args()

    if args.device == 'cpu':
        print('hello?')
        try:
            import spconv.pytorch as spconv
            from spconv.pytorch.conv import SubMConv3d, SparseConv3d
            from functools import partial

            # Monkey-patch the spconv convolution classes to default to ConvAlgo.Native for CPU
            # This ensures that any instance of these classes created later will use the CPU-compatible algorithm.
            SubMConv3d = partial(SubMConv3d, algo=spconv.ConvAlgo.Native)
            SparseConv3d = partial(SparseConv3d, algo=spconv.ConvAlgo.Native)
            
            # In some versions, you might need to re-assign it back to the library's namespace
            # This is a bit more aggressive but can be necessary.
            spconv.SubMConv3d = SubMConv3d
            spconv.SparseConv3d = SparseConv3d
            
            print("INFO: spconv convolution algorithm globally set to 'Native' for CPU compatibility.")

        except ImportError:
            print("WARNING: spconv library not found. Assuming it's not needed.")
            pass

    t0 = time.time()
    inference_single_file(args)
    t1 = time.time()
    print("time taken to inference", (t1-t0), "secs")