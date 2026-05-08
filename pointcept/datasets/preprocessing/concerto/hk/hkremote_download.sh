#!/bin/bash

# Export variables so they can be accessed by subshells spawned by xargs
export BASE_URL="https://data11.map.gov.hk/api/3d-zip/OBJ/"
# To get API_KEY, visit https://3d.map.gov.hk/download-api. After you agree the license, your API_KEY will show in Example URL: https://download.map.gov.hk/api/3d-zip/[format]/[tile_sheet_num].zip?key=[API_KEY]
export API_KEY=${API_KEY}
export OUTPUT_DIR="hk_3d_maps"
export MAX_THREADS=8  # Set the number of concurrent downloads here

mkdir -p "$OUTPUT_DIR"
echo "Files will be downloaded to: $OUTPUT_DIR"
echo "Using $MAX_THREADS concurrent threads..."

# Define the download logic as a function
download_map() {
  local MAP_INDEX=$1
  local ZIP_FILENAME="${MAP_INDEX}.zip"
  local OUTPUT_PATH="${OUTPUT_DIR}/${ZIP_FILENAME}"
  local DOWNLOAD_URL="${BASE_URL}${ZIP_FILENAME}?key=${API_KEY}"

  # Check existence
  if [ -f "$OUTPUT_PATH" ]; then
    echo "[SKIP] File already exists: ${ZIP_FILENAME}"
    return 0
  fi

  echo "[START] Downloading: ${MAP_INDEX}"
  
  # Use wget to download
  # Added --timeout=10 and --tries=2 to prevent threads from hanging forever
  wget --no-check-certificate -q --timeout=10 --tries=2 -O "$OUTPUT_PATH" "$DOWNLOAD_URL"

  # Check wget exit code; $? == 0 means success
  if [ $? -eq 0 ]; then
    echo "[SUCCESS] Downloaded: ${ZIP_FILENAME}"
  else
    echo "[FAILED] Not found or failed: ${ZIP_FILENAME}"
    # Delete the potentially empty file created by a failed download
    rm -f "$OUTPUT_PATH"
  fi
}

# Export the function so xargs subshells can use it
export -f download_map

# Generate the list of map indices and pipe them to xargs for parallel execution
# We use standard space-separated lists for the string variables for cleaner execution
(
  for base in {1..15}; do
    for quad_10k in NW NE SW SE; do
      for grid_2k in {1..25}; do
        for sub_grid_1k in A B C D; do
          echo "${base}-${quad_10k}-${grid_2k}${sub_grid_1k}"
        done
      done
    done
  done
) | xargs -P "$MAX_THREADS" -I {} bash -c 'download_map "$@"' _ {}

echo "All download tasks completed!"