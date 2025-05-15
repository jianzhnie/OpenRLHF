#!/bin/bash

# 加载 CANN 环境变量（路径需根据实际安装位置调整）
# install_path=/usr/local/Ascend
install_path=/root/llmtuner/Ascend
source $install_path/ascend-toolkit/set_env.sh
source $install_path/nnal/atb/set_env.sh

## Robin 环境变量
source /root/llmtuner/miniconda3/bin/activate openrlhf_vllm

# VLLM 相关
# export VLLM_USE_V1=1

# Torch 相关
export ASCEND_LAUNCH_BLOCKING=0
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1


# 以支持torch2.5以上版本
export TORCH_COMPILE_DEBUG=1
export TORCHDYNAMO_DISABLE=1

# ATB 相关
export ATB_LLM_BENCHMARK_ENABLE=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=false
export ATB_LLM_LCOC_ENABLE=false

# HCCL相关
export HCCL_BUFFSIZE=512
export HCCL_DETERMINISTIC=true
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export HCCL_CONNECT_TIMEOUT=1800


export CUDA_DEVICE_MAX_CONNECTIONS=1
