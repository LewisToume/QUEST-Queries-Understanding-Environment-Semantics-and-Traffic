#!/bin/bash

echo "=========================================="
echo "自动驾驶 3D 感知模型环境配置脚本"
echo "RTX 5060 Laptop (CUDA 12.4)"
echo "=========================================="

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 创建虚拟环境
echo "正在创建 conda 环境..."
conda create -n autodrive_3d python=3.10 -y

# 激活环境
echo "激活环境..."
eval "$(conda shell.bash hook)"
conda activate autodrive_3d

# 安装 PyTorch (CUDA 12.4)
echo "安装 PyTorch 2.x with CUDA 12.4 支持..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装 transformers
echo "安装 transformers..."
pip install transformers>=4.30.0

# 安装 flash-attn (需要编译)
echo "安装 flash-attn (可能需要几分钟)..."
pip install flash-attn --no-build-isolation

# 安装 spconv (CUDA 12.4 版本)
echo "安装 spconv for CUDA 12.4..."
pip install spconv-cu124 -f https://github.com/Find_DEFINITION/spconv-wheels/releases

# 安装其他必备库
echo "安装其他依赖库..."
pip install timm numpy opencv-python Pillow scipy tqdm

echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate autodrive_3d"
echo "  python backbone.py"
echo ""
