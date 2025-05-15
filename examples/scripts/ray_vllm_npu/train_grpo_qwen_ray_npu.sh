#!/bin/bash

set -x

work_dir="/root/llmtuner/llm/OpenRLHF-npu/work_dir/grpo/qwen25_7b/"
# 如果work_dir不存，则创建该目录
if [ ! -d "${work_dir}" ]; then
    mkdir -p ${work_dir}
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/root/llmtuner/llm/OpenRLHF-npu/work_dir"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.8 \
   --pretrain /root/llmtuner/hfhub/models/Qwen/Qwen2.5-7B \
   --remote_rm_url /root/llmtuner/llm/OpenRLHF-npu/openrlhf/trainer/ppo_utils/math_reward_fn.py \
   --save_path ${work_dir} \
   --ckpt_path ${work_dir}/ckpt \
   --save_hf_ckpt \
   --max_ckpt_num 100 \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 4 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --num_episodes 3 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --gamma 1.0 \
   --init_kl_coef 0 \
   --advantage_estimator group_norm \
   --system_prompt qwen_math_cot \
   --policy_loss_fn drgrpo \
   --eps_clip_high 0.28 \
   --correct_reward 1.0 \
   --incorrect_reward 0.0 \
   --prompt_data /root/llmtuner/hfhub/datasets/math_lvl3to5_8k/train.jsonl \
   --dataset_config default \
   --input_key problem \
   --label_key answer \
   --max_samples 100000 \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb 8ad58a961091cd50e8ed0b00d9060e502209a2b5 \
   --wandb_project openrlhf-vlm-ray-grpo \
   --wandb_group grpo \
   --wandb_run_name Qwen2.5-7B-grpo-baseline

# You could also try
#   --kl_estimator k2 \
