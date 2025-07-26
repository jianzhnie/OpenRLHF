from typing import List
from collections import defaultdict
import math
import numpy as np
from transformers import PreTrainedTokenizer
from openrlhf.trainer.ppo_utils.math_reward_funcs import MathAccuracyReward, BaseRewardFunction


class GRPOLambdaReward(BaseRewardFunction):
    """
    Implements GRPO-λ dynamic reward assignment based on group-wise accuracy ranking.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for measuring response length.
        math_accuracy_reward (BaseRewardFunction): Reward function for accuracy-prioritized groups (0/1 reward).
        accuracy_top_lambda (float): Proportion of top-accurate groups to assign efficiency reward.
        length_penalty_alpha (float): Length penalty strength.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        math_acc_reward: MathAccuracyReward,
        accuracy_top_lambda: float = 0.2,
        length_penalty_alpha: float = 0.6,  # 长度惩罚强度
    ) -> None:
        self.tokenizer = tokenizer
        self.math_acc_reward = math_acc_reward
        self.accuracy_top_lambda = accuracy_top_lambda
        self.length_penalty_alpha = length_penalty_alpha

    def __call__(self, completions: List[str], solutions: List[str], groups: List[int], **kwargs) -> List[float]:
        """
        Computes the rewards for each completion based on group-wise accuracy and length penalties.

        Args:
            completions (List[str]): Generated completions.
            solutions (List[str]): Ground truth answers.
            groups (List[int]): Group ID for each completion.

        Returns:
            List[float]: Reward for each completion.
        """
        assert len(completions) == len(solutions) == len(groups), "Input lengths must match"
        # Step 1: Compute token lengths for each completion

        lengths = [len(ids) for ids in self.tokenizer(completions, add_special_tokens=False)["input_ids"]]

        # Step 2: Compute accuracy for each sample
        acc_rewards = self.math_acc_reward(completions, solutions, **kwargs)
        is_correct = [r >= 1.0 for r in acc_rewards]

        # Step 3: Group-wise accuracy and length statistics
        group_correct_count = defaultdict(int)
        group_total_count = defaultdict(int)
        group_lengths = defaultdict(list)

        for g, correct, length in zip(groups, is_correct, lengths):
            group_total_count[g] += 1
            if correct:
                group_correct_count[g] += 1
                group_lengths[g].append(length)

        # Step 4: Compute accuracy per group
        group_accuracy = {g: group_correct_count[g] / group_total_count[g] for g in group_total_count}

        # Step 5: Sort groups by accuracy, select top λ%
        sorted_groups = sorted(group_accuracy.items(), key=lambda x: x[1], reverse=True)
        num_top_groups = max(1, math.ceil(len(sorted_groups) * self.accuracy_top_lambda))
        top_group_ids = {g for g, _ in sorted_groups[:num_top_groups]}

        # Step 6: Calculate mean and std of lengths for correct answers in each group
        group_length_stats = {}
        for g in group_lengths:
            if group_lengths[g]:
                mean_length = np.mean(group_lengths[g])
                std_length = (
                    np.std(group_lengths[g]) if len(group_lengths[g]) > 1 else 0.0001
                )  # Avoid division by zero
                group_length_stats[g] = (mean_length, std_length)

        # Step 7: Apply corresponding reward function
        rewards = []
        for i, (accuracy, length, group) in enumerate(zip(acc_rewards, lengths, groups)):
            if group in top_group_ids:
                # Efficiency-prioritized group: use efficiency reward
                mean_length, std_length = group_length_stats[group]
                sigma_value = (length - mean_length) / std_length
                reward = 1 - self.length_penalty_alpha * self.sigmoid(sigma_value)
                reward = reward * accuracy
            else:
                # Accuracy-prioritized group: use accuracy reward
                reward = acc_rewards[i]
            rewards.append(reward)

        return rewards

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid function."""
        return 1 / (1 + math.exp(-x))
