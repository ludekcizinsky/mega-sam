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
RAFT_CKPT=/scratch/izar/cizinsky/pretrained/raft-things.pth
OUT_DIR=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$scene_name/megasam/

# Run Raft Optical Flows
CUDA_VISIBLE_DEVICES=0 python cvd_opt/preprocess_flow.py \
--datapath=$DATA_DIR \
--model=$RAFT_CKPT \
--outdir $OUT_DIR \
--scene_name $scene_name --mixed_precision

# Run CVD optmization
CUDA_VISIBLE_DEVICES=0 python cvd_opt/cvd_opt.py \
--scene_name $scene_name \
--output_dir $OUT_DIR \
--w_grad 2.0 --w_normal 5.0 

