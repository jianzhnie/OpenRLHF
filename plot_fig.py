import matplotlib.pyplot as plt

# 定义三个模型的数据
base_data = {"steps": [0], "scores": [35.8], "label": "Base Model"}
sft_data = {"steps": [0], "scores": [49.4], "label": "With SFT"}
rl_data = {
    "steps": [5 * i for i in range(1, 15)],  # 5到70步（间隔5）
    "scores": [36.8, 44.8, 55.8, 60.6, 64.8, 64.4, 67.4, 67.4, 66.0, 67.8, 68.8, 67.4, 68.2, 67.0],
    "label": "With RL",
}

# 合并所有数据并按步骤排序
all_steps = sorted(set(base_data["steps"] + sft_data["steps"] + rl_data["steps"]))
all_scores = []
all_labels = []

# 添加Base/SFT数据（仅在0步）
for step in [0]:
    all_steps.append(step)
    all_scores.append(base_data["scores"][0])
    all_labels.append(base_data["label"])
    all_scores.append(sft_data["scores"][0])
    all_labels.append(sft_data["label"])

# 添加RL数据（跳过重复的0步）
rl_index = 0
for step in rl_data["steps"]:
    if step not in all_steps:
        all_steps.append(step)
        all_scores.append(rl_data["scores"][rl_index])
        all_labels.append(rl_data["label"])
    rl_index += 1

# 创建画布
plt.figure(figsize=(14, 7))

# 绘制三条曲线
plt.plot(base_data["steps"], base_data["scores"], linestyle="--", color="#2ca02c", marker="o", label="Base Model")
plt.plot(sft_data["steps"], sft_data["scores"], linestyle="-.", color="#ff7f0e", marker="s", label="With SFT")
plt.plot(rl_data["steps"], rl_data["scores"], linestyle="-", color="#1f77b4", marker="o", label="With RL")

# 添加标题和标签
plt.title("Model Comparison: RL vs SFT vs Base (Qwen2.5-1.5B-Math)", fontsize=14)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Model Performance on Math500", fontsize=12)
plt.legend(fontsize=10)

# # 数据标签（仅显示关键点）
for i in range(len(all_steps)):
    if all_steps[i] == 0:
        # 基线和SFT在0步重叠，合并标签
        plt.text(
            all_steps[i],
            (all_scores[i] + all_scores[i + 1]) / 2,
            f"Base: {all_scores[i]:.1f}\nSFT: {all_scores[i+1]:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#666",
        )
        i += 1  # 跳过下一个重复的0步数据
    else:
        plt.text(all_steps[i], all_scores[i], f"{all_scores[i]:.1f}", ha="center", va="bottom", fontsize=9)

# 网格线
plt.grid(linestyle="--", alpha=0.6)

# 保存矢量图
plt.savefig("model_comparison.png")

# 展示图表
plt.show()
