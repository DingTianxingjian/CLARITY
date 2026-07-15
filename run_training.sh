#!/bin/bash
# CLARITY/run_training.sh — train exp012 (BrainIAC online encoder)
#
# Before running, place the following external files alongside this repo:
#
#   BrainIAC-main/src/checkpoints/BrainIAC.ckpt        ← BrainIAC ViT-B weights
#   Predictor/dataset/MU_Glioma_Post/features_output.csv   ← pre-extracted features
#   Predictor/dataset/MU_Glioma_Post/clinical_latest.json  ← clinical timeline
#   datasets/MU-Glioma-Post/                               ← raw MRI NIfTI files
#
# Usage:
#   cd CLARITY && bash run_training.sh
#   bash run_training.sh --resume <path/to/checkpoint.pth>
#   bash run_training.sh --exp_name my_experiment

set -e

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BRAINIAC_CKPT="BrainIAC-main/src/checkpoints/BrainIAC.ckpt"
FEATURES_CSV="Predictor/dataset/MU_Glioma_Post/features_output.csv"
TIMELINE_JSON="Predictor/dataset/MU_Glioma_Post/clinical_latest.json"
MRI_DATA_DIR="datasets/MU-Glioma-Post"
EXP_NAME="exp012_cf_diversity"
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --resume)    RESUME_FROM="$2"; shift 2 ;;
    --exp_name)  EXP_NAME="$2";   shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "=========================================="
echo "CLARITY — Training exp012 (BrainIAC)"
echo "  exp_name : $EXP_NAME"
echo "  resume   : ${RESUME_FROM:-<scratch>}"
echo "=========================================="

for path in "$BRAINIAC_CKPT" "$FEATURES_CSV" "$TIMELINE_JSON" "$MRI_DATA_DIR"; do
  if [ ! -e "$path" ]; then
    echo ""
    echo "ERROR: required path not found: $path"
    echo "See the header of this script for data placement instructions."
    exit 1
  fi
done
echo "Data paths OK."
echo ""

export PYTHONPATH="$SCRIPT_DIR/Predictor:$SCRIPT_DIR"

RESUME_ARG=""
[ -n "$RESUME_FROM" ] && RESUME_ARG="--resume_from $RESUME_FROM"

$PYTHON Predictor/train.py \
  --exp_name          "$EXP_NAME" \
  --features_csv      "$FEATURES_CSV" \
  --timeline_json     "$TIMELINE_JSON" \
  --mri_data_dir      "$MRI_DATA_DIR" \
  --brainiac_ckpt     "$BRAINIAC_CKPT" \
  --brainiac_tokens   8 \
  --brainiac_lora_r   8 \
  --latent_dim        768 \
  --num_modalities    32 \
  --take_dims         767 \
  --text_encoder_name google/medgemma-4b-it \
  --num_epochs        100 \
  --batch_size        16 \
  --val_batch_size    16 \
  --num_workers       4 \
  --lr                2e-4 \
  --text_lr           1e-4 \
  --text_proj_lr      5e-4 \
  --cf_weight         1.0 \
  --cf_cos_margin     0.9 \
  --lambda_l1         0.5 \
  --lambda_cox        1.0 \
  --lambda_bce        1.0 \
  --dropout           0.3 \
  --grad_clip_norm    2.0 \
  --survival_wd       0.01 \
  --seed              42 \
  $RESUME_ARG

echo ""
echo "Training complete."
