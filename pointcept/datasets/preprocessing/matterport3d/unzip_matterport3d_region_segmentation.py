import argparse
import os
import zipfile
import glob
import time


def unzip_file(input_path, output_path):
    with zipfile.ZipFile(input_path, "r") as zip_ref:
        zip_ref.extractall(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unzip all "region_segmentations.zip" files in a directory'
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to input directory containing ZIP files",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory for extracted files",
        required=True,
    )
    args = parser.parse_args()

    data = args.data
    output_dir = args.output_dir

    for filename in glob.glob(os.path.join(data, "*", "region_segmentations.zip")):
        input_path = os.path.join(data, filename)
        print(f"Extracting {input_path} to {output_dir}...")
        unzip_file(input_path, output_dir)
