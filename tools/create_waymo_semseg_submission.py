"""
Script for Creating Waymo Semantic Segmentation Submission

The Waymo dataset toolkit relies on an old version of Tensorflow
which share a conflicting dependency with the Pointcept environment,
therefore we detach the submission generation from the test process
and the script require the following environment:

```bash
conda create -n waymo python=3.8 -y
conda activate waymo
pip3 install waymo-open-dataset-tf-2-11-0
```

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import tqdm
import argparse
import numpy as np
import zlib
import waymo_open_dataset.dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2


def compress_array(array: np.ndarray, is_int32: bool = False):
    """Compress a numpy array to ZLIP compressed serialized MatrixFloat/Int32.

    Args:
      array: A numpy array.
      is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

    Returns:
      The compressed bytes.
    """
    if is_int32:
        m = open_dataset.MatrixInt32()
    else:
        m = open_dataset.MatrixFloat()
    m.shape.dims.extend(list(array.shape))
    m.data.extend(array.reshape([-1]).tolist())
    return zlib.compress(m.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record_path",
        required=True,
        help="Path to the prediction result folder of Waymo dataset",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the processed Waymo dataset",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["validation", "testing"],
        help="Split of the prediction ([training, validation, testing]).",
    )
    args = parser.parse_args()
    file_list = [file for file in os.listdir(args.record_path) if file.endswith(".npy")]
    submission = segmentation_submission_pb2.SemanticSegmentationSubmission()
    frames = segmentation_metrics_pb2.SegmentationFrameList()
    bar = tqdm.tqdm(file_list)
    for file in bar:
        bar.set_postfix(file=file)
        context_name, frame_timestamp_micros = file.strip("segment-*_pred.npy").split(
            "_with_camera_labels_"
        )
        # Load prediction.
        # In Pointcept waymo dataset, we minus 1 to label to ignore UNLABELLED class (0 -> -1)
        pred = np.load(os.path.join(args.record_path, file)) + 1
        masks = np.load(
            os.path.join(
                args.dataset_path,
                args.split,
                f"segment-{context_name}_with_camera_labels",
                frame_timestamp_micros,
                "mask.npy",
            ),
            allow_pickle=True,
        )
        offset = np.cumsum([mask.sum() for mask in masks.reshape(-1)])
        pred = np.split(pred[: offset[-1]], offset[:-1])
        pred_ri1 = pred[0]
        pred_ri2 = pred[5]
        mask_ri1 = np.expand_dims(masks[0, 0], -1)
        mask_ri2 = np.expand_dims(masks[1, 0], -1)
        range_dummy = np.zeros_like(mask_ri1, dtype=np.int32)
        range_pred_ri1 = np.zeros_like(mask_ri1, dtype=np.int32)
        range_pred_ri1[mask_ri1] = pred_ri1
        range_pred_ri1 = np.concatenate([range_dummy, range_pred_ri1], axis=-1)
        range_pred_ri2 = np.zeros_like(mask_ri2, dtype=np.int32)
        range_pred_ri2[mask_ri2] = pred_ri2
        range_pred_ri2 = np.concatenate([range_dummy, range_pred_ri2], axis=-1)

        # generate frame submission
        segmentation_label = open_dataset.Laser()
        segmentation_label.name = open_dataset.LaserName.TOP
        segmentation_label.ri_return1.segmentation_label_compressed = compress_array(
            range_pred_ri1, is_int32=True
        )
        segmentation_label.ri_return2.segmentation_label_compressed = compress_array(
            range_pred_ri2, is_int32=True
        )
        frame = segmentation_metrics_pb2.SegmentationFrame()
        frame.segmentation_labels.append(segmentation_label)
        frame.context_name = context_name
        frame.frame_timestamp_micros = int(frame_timestamp_micros)
        frames.frames.append(frame)
    submission.account_name = "***"
    submission.unique_method_name = "***"
    submission.authors.append("***")
    submission.affiliation = "***"
    submission.method_link = "***"
    submission.sensor_type = (
        segmentation_submission_pb2.SemanticSegmentationSubmission.LIDAR_ALL
    )
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(frames)
    output_filename = os.path.join(args.record_path, "submission.bin")
    f = open(output_filename, "wb")
    f.write(submission.SerializeToString())
    f.close()
