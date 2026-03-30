#!/bin/bash
# 训练启动脚本

set -e  # 遇到错误立即停止

echo "=========================================="
echo "🚀 启动胶质瘤生存预测模型训练"
echo "=========================================="
echo ""

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU训练（会很慢）"
    echo ""
fi

# 检查数据文件
echo "📁 检查数据文件..."
if [ ! -f "Predictor/dataset/MU_Glioma_Post/clinical_latest.json" ]; then
    echo "❌ 缺少: Predictor/dataset/MU_Glioma_Post/clinical_latest.json"
    exit 1
fi
if [ ! -f "BrainIAC-main/src/checkpoints/BrainIAC.ckpt" ]; then
    echo "❌ 缺少: BrainIAC-main/src/checkpoints/BrainIAC.ckpt"
    exit 1
fi
if [ ! -d "datasets/MU-Glioma-Post" ]; then
    echo "❌ 缺少MRI目录: datasets/MU-Glioma-Post"
    exit 1
fi
echo "✅ 数据文件完整"
echo ""

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "🎯 启动MRI backbone训练..."
echo ""
python Predictor/train.py \
    --exp_name mri_backbone_$(date +%Y%m%d_%H%M%S) \
    --mri_size 96 \
    --contrastive_weight 0.1 \
    --vision_checkpoint ./BrainIAC-main/src/checkpoints/BrainIAC.ckpt

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
