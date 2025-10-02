#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

TORCH_HOME="/scratch/izar/cizinsky/.cache"
HF_HOME="/scratch/izar/cizinsky/.cache"

scene_name="football_high_res"
DATA_DIR=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$scene_name/image
OUT_DIR=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$scene_name/megasam
mkdir -p $OUT_DIR

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc cuda/11.8 ffmpeg
conda activate mega_sam

# Run DepthAnything
CUDA_VISIBLE_DEVICES=0 python Depth-Anything/run_videos.py --encoder vitl \
--load-from /scratch/izar/cizinsky/pretrained/depth_anything_vitl14.pth \
--img-path $DATA_DIR \
--outdir $OUT_DIR/depth_anything

# Run UniDepth
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

CUDA_VISIBLE_DEVICES=0 python UniDepth/scripts/demo_mega-sam.py \
--scene-name $scene_name \
--img-path $DATA_DIR \
--outdir $OUT_DIR/unidepth
