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
   --pretrain /root/llmtuner/hfhub/models/Qwen/Qwen2.5-Math-1.5B \
   --reward_func_names accuracy format reasoning_steps cosine \
   --save_path ./${work_dir}/qwen2.5-1.5b \
   --ckpt_path ./${work_dir}/qwen2.5-1.5b/ckpt \
   --save_hf_ckpt \
   --max_ckpt_num 1000 \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --num_episodes 3 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --value_clip 0.2 \
   --gamma 1.0 \
   --init_kl_coef 1e-3 \
   --kl_estimator k3 \
   --kl_clip_max 10 \
   --advantage_estimator group_norm \
   --cosine_min_value_wrong -0.5   \
   --cosine_max_value_wrong 0.0   \
   --cosine_min_value_correct 0.6   \
   --cosine_max_value_correct 1.0   \
   --cosine_max_len 1024 \
   --prompt_data /root/llmtuner/hfhub/datasets/AI-MO/NuminaMath-TIR \
   --input_key problem \
   --label_key solution \
   --apply_chat_template \
   --max_samples 100000 \
   --normalize_reward \
   --adam_offload \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_tensorboard  ./${work_dir}/qwen2.5-1.5b/tb_log
EOF

    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    nohup deepspeed --module $training_commands > ${work_dir}/train.log 2>&1 &
fi