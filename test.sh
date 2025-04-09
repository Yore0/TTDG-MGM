#!/bin/bash

# 配置环境变量
CUDA_VISIBLE_DEVICES=7

# 定义基础命令和文件路径
BASE_COMMAND="python train_net.py --eval-only --config configs/test_segment.yaml MODEL.WEIGHTS"
BASE_PATH="output/Fundus/Drishti_sin/model_000"

# 遍历 x 的范围 0-9
for x in {0..9}; do
    # 构建完整的文件路径
    FILE_PATH="${BASE_PATH}${x}999.pth"
    
    # 构建完整的命令
    FULL_COMMAND="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${BASE_COMMAND} ${FILE_PATH}"
    
    # 输出运行的命令（可选）
    echo "Running: ${FULL_COMMAND}"
    
    # 运行命令
    eval "${FULL_COMMAND}"
done
