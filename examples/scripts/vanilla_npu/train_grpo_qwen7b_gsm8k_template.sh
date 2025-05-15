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
   --reward_func_names accuracy \
   --save_path ./${work_dir}/Qwen2.5-7B \
   --ckpt_path ./${work_dir}/Qwen2.5-7B/ckpt \
   --save_hf_ckpt \
   --max_ckpt_num 1000 \
   --save_steps 5 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 16 \
   --num_episodes 3 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1200 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --value_clip 0.2 \
   --gamma 1.0 \
   --init_kl_coef 1e-3 \
   --kl_estimator k3 \
   --kl_clip_max 10 \
   --advantage_estimator group_norm \
   --prompt_data /root/llmtuner/hfhub/datasets/openai/gsm8k \
   --dataset_config main \
   --input_key question \
   --label_key answer \
   --apply_chat_template \
   --max_samples 100000 \
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