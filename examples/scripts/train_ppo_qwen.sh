set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain /home/robin/hf_hub/models/Qwen/Qwen1.5-0.5B \
   --critic_pretrain /home/robin/hf_hub/models/Qwen/Qwen1.5-0.5B \
   --reward_func_names accuracy format reasoning_steps cosine \
   --save_path ./work_dir/qwen2.5-1.5b-rlhf \
   --ckpt_path ./work_dir/qwen2.5-1.5b-rlhf/ckpt \
   --save_hf_ckpt \
   --max_ckpt_num 100 \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 16 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --value_clip 0.2 \
   --init_kl_coef 0.01 \
   --kl_target 6 \
   --kl_horizon 10000 \
   --prompt_data /home/robin/hf_hub/datasets/text_data/AI-MO/NuminaMath-TIR \
   --input_key problem \
   --label_key solution \
   --apply_chat_template \
   --max_samples 100000 \
   --normalize_reward \
   --adam_offload \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_tensorboard  ./work_dir/qwen2.5-1.5b-rlhf/tb_log
EOF

    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
