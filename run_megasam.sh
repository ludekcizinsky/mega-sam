#!/bin/bash

# ---- Setup of paths (edit as needed) and conda env
# Set cache directories for models so you don't fill up home directory
TORCH_HOME="/scratch/izar/cizinsky/.cache"
HF_HOME="/scratch/izar/cizinsky/.cache"

# Set scene name and directories (data dir is where you have your frames, our dir is where ALL outputs will go)
scene_name="modric_vs_ribberi"
DATA_DIR=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$scene_name/image
OUT_DIR=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$scene_name/megasam
mkdir -p $OUT_DIR
# (Derived paths for intermediate outputs no need to do anything)
MONO_DEPTH_PATH=$OUT_DIR/depth_anything
METRIC_DEPTH_PATH=$OUT_DIR/unidepth

# Set checkpoints for downloaded pretrained models (mega sam is already in the repo, others you need to download)
MEGASAM_CKPT=checkpoints/megasam_final.pth 
RAFT_CKPT=/scratch/izar/cizinsky/pretrained/raft-things.pth
DEPTH_ANY_CKPT=/scratch/izar/cizinsky/pretrained/depth_anything_vitl14.pth

# Activate conda environment (make sure you source conda.sh from your install location)
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate mega_sam

# --- Run the full pipeline (no need to change anything below here unless you want to modify parameters)
# Run DepthAnything
CUDA_VISIBLE_DEVICES=0 python Depth-Anything/run_videos.py --encoder vitl \
--load-from $DEPTH_ANY_CKPT \
--img-path $DATA_DIR \
--outdir $OUT_DIR/depth_anything

# Run UniDepth
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"
CUDA_VISIBLE_DEVICES=0 python UniDepth/scripts/demo_mega-sam.py \
--scene-name $scene_name \
--img-path $DATA_DIR \
--outdir $OUT_DIR/unidepth

# Run camera tracking
CUDA_VISIBLE_DEVICE=0 python camera_tracking_scripts/test_demo.py \
--datapath=$DATA_DIR \
--weights=$MEGASAM_CKPT \
--scene_name $scene_name \
--mono_depth_path $MONO_DEPTH_PATH \
--metric_depth_path $METRIC_DEPTH_PATH \
--outdir $OUT_DIR

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
--w_grad 2.0 \
--w_normal 5.0