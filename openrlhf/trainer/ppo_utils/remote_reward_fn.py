from typing import List
import torch
from openrlhf.trainer.ppo_utils.math_reward_funcs import MathAccuracyReward

reward_fn = MathAccuracyReward()


def reward_func(completions, solutions) -> List[float]:
    rewards = reward_fn(completions, solutions)
    return torch.tensor(rewards)
