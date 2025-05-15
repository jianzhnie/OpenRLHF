#!/bin/bash

project_name="openrlhf-qwen2-5-1.5b_NuminaMath-TIR"

tflogs=(
    # "/root/llmtuner/llm/OpenRLHF/work_dir/grpo_qwen-math1.5b_NuminaMtah-TIR/20250312_110258/qwen2.5-1.5b/tb_log/ppo_0312T11:03"
    # "/root/llmtuner/llm/OpenRLHF/work_dir/grpo_qwen-math1.5b_NuminaMtah-TIR/20250312_111344/qwen2.5-1.5b/tb_log/ppo_0312T11:14"
    # "/root/llmtuner/llm/OpenRLHF/work_dir/grpo_qwen-math1.5b_NuminaMtah-TIR/20250312_185319/qwen2.5-1.5b/tb_log/ppo_0312T18:53"
    # "/root/llmtuner/llm/OpenRLHF/work_dir/grpo_qwen-math1.5b_NuminaMtah-TIR/20250319_152738/qwen2.5-1.5b/tb_log/ppo_0319T15:27"
    "/root/llmtuner/llm/OpenRLHF/work_dir/ppo/20250312_131405/qwen2.5-1.5b/tb_log/ppo_0312T13:14"
)

for logdir in "${tflogs[@]}"; do
    echo "Looking for event files in: $logdir"
    for event_file in "$logdir"/events.out.tfevents.*; do
        if [[ -f "$event_file" ]]; then
            echo "Syncing $event_file to Weights & Biases under project $project_name..."
            WANDB_PROJECT="$project_name" wandb sync "$event_file"
        else
            echo "No event files found in $logdir"
        fi
    done
done

