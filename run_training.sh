#!/bin/bash
# CLARITY/run_training.sh — reproduce the released CLARITY model (BrainIAC online encoder + MedGemma)
#
# Training runs in two stages, matching how the reported model was trained:
#   Stage 1 (exp011_cond_in_survival): 60 epochs, 3-epoch L1-only warmup, no counterfactual loss
#   Stage 2 (exp012_cf_diversity)    : resume from stage-1 best_c_index.pth, add the
#                                      counterfactual diversity loss, continue to epoch 100
# Both stages use an EMA target encoder for BrainIAC (momentum 0.996).
#
# Required files (place alongside this repo):
#
#   BrainIAC-main/src/checkpoints/BrainIAC.ckpt        ← BrainIAC ViT-B weights
#   Predictor/dataset/MU_Glioma_Post/clinical_latest.json  ← clinical timeline (required)
#   datasets/MU-Glioma-Post/                               ← raw MRI NIfTI files (required)
#   Access to google/medgemma-4b-it on HuggingFace (gated; run `huggingface-cli login`)
#
# Optional:
#   Predictor/dataset/MU_Glioma_Post/features_output.csv   ← pre-extracted BrainIAC features
#     If present, used as a data-availability filter (only patients in the CSV are trained on).
#     If absent, all patients in clinical_latest.json with MRI files are used.
#
# Usage:
#   cd CLARITY && bash run_training.sh              # stage 1 then stage 2
#   bash run_training.sh --stage 1                  # stage 1 only
#   bash run_training.sh --stage 2                  # stage 2 only (needs stage-1 checkpoint)
#   bash run_training.sh --stage1_name A --stage2_name B

set -e

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BRAINIAC_CKPT="BrainIAC-main/src/checkpoints/BrainIAC.ckpt"
FEATURES_CSV="Predictor/dataset/MU_Glioma_Post/features_output.csv"
TIMELINE_JSON="Predictor/dataset/MU_Glioma_Post/clinical_latest.json"
MRI_DATA_DIR="datasets/MU-Glioma-Post"
STAGE="all"
STAGE1_NAME="exp011_cond_in_survival"
STAGE2_NAME="exp012_cf_diversity"

while [[ $# -gt 0 ]]; do
  case $1 in
    --stage)        STAGE="$2";       shift 2 ;;
    --stage1_name)  STAGE1_NAME="$2"; shift 2 ;;
    --stage2_name)  STAGE2_NAME="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

STAGE1_CKPT="experiments/$STAGE1_NAME/checkpoints/best_c_index.pth"

echo "=========================================="
echo "CLARITY — Training (stage: $STAGE)"
echo "  stage 1 : $STAGE1_NAME"
echo "  stage 2 : $STAGE2_NAME"
echo "=========================================="

for path in "$BRAINIAC_CKPT" "$TIMELINE_JSON" "$MRI_DATA_DIR"; do
  if [ ! -e "$path" ]; then
    echo ""
    echo "ERROR: required path not found: $path"
    echo "See the header of this script for data placement instructions."
    exit 1
  fi
done

# features_csv is optional: pass it only if the file exists
FEATURES_CSV_ARG=""
if [ -f "$FEATURES_CSV" ]; then
  FEATURES_CSV_ARG="--features_csv $FEATURES_CSV"
  echo "features_csv : $FEATURES_CSV (found)"
else
  echo "features_csv : not found — using all patients from clinical_latest.json"
fi
echo "Data paths OK."
echo ""

export PYTHONPATH="$SCRIPT_DIR/Predictor:$SCRIPT_DIR"

# Arguments shared by both stages
COMMON_ARGS=(
  $FEATURES_CSV_ARG
  --timeline_json         "$TIMELINE_JSON"
  --mri_data_dir          "$MRI_DATA_DIR"
  --brainiac_ckpt         "$BRAINIAC_CKPT"
  --brainiac_tokens       8
  --brainiac_lora_r       8
  --brainiac_ema_momentum 0.996
  --latent_dim            768
  --num_modalities        32
  --take_dims             767
  --text_encoder_name     google/medgemma-4b-it
  --batch_size            16
  --val_batch_size        16
  --num_workers           4
  --lr                    2e-4
  --text_lr               1e-4
  --text_proj_lr          5e-4
  --vision_lr             5e-5
  --lambda_l1             0.5
  --lambda_cox            1.0
  --lambda_bce            1.0
  --dropout               0.3
  --grad_clip_norm        2.0
  --survival_wd           0.01
  --contrastive_weight    0.0
  --variance_weight       0.0
  --early_stopping_patience 0
  --seed                  42
)

if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
  echo ">>> Stage 1: $STAGE1_NAME"
  $PYTHON Predictor/train.py \
    "${COMMON_ARGS[@]}" \
    --exp_name      "$STAGE1_NAME" \
    --num_epochs    60 \
    --warmup_epochs 3 \
    --cf_weight     0.0
fi

if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
  if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: stage-1 checkpoint not found: $STAGE1_CKPT"
    exit 1
  fi
  echo ">>> Stage 2: $STAGE2_NAME (resume from $STAGE1_CKPT)"
  $PYTHON Predictor/train.py \
    "${COMMON_ARGS[@]}" \
    --exp_name      "$STAGE2_NAME" \
    --num_epochs    100 \
    --warmup_epochs 0 \
    --cf_weight     1.0 \
    --cf_cos_margin 0.9 \
    --resume_from   "$STAGE1_CKPT"
fi

echo ""
echo "Training complete."
