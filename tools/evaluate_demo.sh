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

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc cuda/11.8 ffmpeg
conda activate mega_sam

TORCH_HOME="/scratch/izar/cizinsky/.cache"
HF_HOME="/scratch/izar/cizinsky/.cache"

scene_name="football_high_res"
DATA_DIR=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$scene_name/image
CKPT_PATH=checkpoints/megasam_final.pth
MONO_DEPTH_PATH=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$scene_name/megasam/depth_anything
METRIC_DEPTH_PATH=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$scene_name/megasam/unidepth
OUT_DIR=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$scene_name/megasam/

CUDA_VISIBLE_DEVICE=0 python camera_tracking_scripts/test_demo.py \
--datapath=$DATA_DIR \
--weights=$CKPT_PATH \
--scene_name $scene_name \
--mono_depth_path $MONO_DEPTH_PATH \
--metric_depth_path $METRIC_DEPTH_PATH \
--outdir $OUT_DIR \