set -x

date=`date '+%Y%m%d_%H%M%S'`

export HCCL_CONNECT_TIMEOUT=1200
work_dir="work_dir/grpo/${date}"
# 如果work_dir不存，则创建该目录
if [ ! -d "./${work_dir}" ]; then
    mkdir -p ./${work_dir}
fi

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain /root/llmtuner/hfhub/models/Qwen/Qwen2.5-7B \
   --reward_func_names accuracy format reasoning_steps cosine repetition \
   --save_path ./${work_dir}/Qwen2.5-7B \
   --ckpt_path ./${work_dir}/Qwen2.5-7B/ckpt \
   --save_hf_ckpt \
   --max_ckpt_num 1000 \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --num_episodes 2 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --generate_max_len 2048 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --value_clip 0.2 \
   --gamma 1.0 \
   --init_kl_coef 0 \
   --advantage_estimator group_norm \
   --correct_reward 1.0 \
   --incorrect_reward 0.0 \
   --cosine_min_value_wrong -0.5   \
   --cosine_max_value_wrong 0.0   \
   --cosine_min_value_correct 0.6   \
   --cosine_max_value_correct 1.0   \
   --cosine_max_len 2048 \
   --prompt_data /root/llmtuner/hfhub/datasets/open-r1/OpenR1-Math-220k \
   --input_key problem \
   --label_key solution \
   --dataset_config default \
   --apply_chat_template \
   --max_samples 1000000 \
   --normalize_reward \
   --adam_offload \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_tensorboard  ./${work_dir}/Qwen2.5-7B/tb_log
EOF

    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    nohup deepspeed --module $training_commands > ${work_dir}/train.log 2>&1 &
fi