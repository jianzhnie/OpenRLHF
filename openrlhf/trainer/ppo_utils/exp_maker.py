import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
from openrlhf.utils.statistics import MovAvg, RunningMeanStd

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: list[str]
    pad_len: Optional[int]


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: Union[list[str], str] = None,
        reward_fn=None,
        reward_func_names: list[str] = None,
        reward_funcs: List[Callable] = None,
        reward_weights: torch.Tensor = None,
        with_reward_baseline: bool = False,
        normalize_rule_reward: bool = False,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.reward_func_names = reward_func_names
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        self.normalize_rule_reward = normalize_rule_reward
        if normalize_rule_reward:
            self.reward_normalizer = RunningMeanStd()

        self.with_reward_baseline = with_reward_baseline
        if with_reward_baseline:
            self.reward_moving_avg = MovAvg(size=1024)

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, bath_inputs, **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(bath_inputs, **generate_kwargs)
                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = torch.distributed.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            samples_list = self.generate_samples(bath_inputs, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, batch_inputs, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = batch_inputs["prompt"]
        all_labels = batch_inputs["label"]
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            labels = all_labels[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                labels=labels,
                pad_len=None,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        else:
            base_action_log_probs = None

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.reward_funcs:
            completion_ids = sequences[:, -num_actions:]
            completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            rewards, reward_func_metrics = self.compute_func_rewards(completions, samples.labels)

        elif self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            if self.custom_reward_func:
                rewards = self.custom_reward_func(queries, samples.prompts, samples.labels).to(
                    device=action_log_probs.device
                )
            else:
                rewards = remote_rm_fn(
                    self.remote_rm_url, queries=queries, prompts=samples.prompts, labels=samples.labels
                ).to(device=action_log_probs.device)
        else:
            # local RM
            rewards = self.reward_model(sequences, attention_mask)

        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
                kl_clip_max=self.strategy.args.kl_clip_max,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": rewards,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.reward_funcs:
            info.update(reward_func_metrics)

        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    def compute_func_rewards(self, completions, solutions, reward_kwargs=None) -> Tuple[torch.Tensor, dict]:
        """
        计算多个自定义奖励函数的值并返回加权组合结果

        参数:
        - completions: 模型生成的完成序列
        - solutions: 参考答案
        - reward_kwargs: 额外的关键字参数

        返回:
        - rewards: 加权组合后的奖励值张量
        - reward_func_metrics: 各奖励函数的指标字典
        """

        # 初始化每个奖励函数的指标字典
        reward_func_metrics = {}
        num_completions = len(completions)
        num_funcs = len(self.reward_funcs)

        # 为每个生成的序列计算每个奖励函数的值
        rewards_per_func = torch.zeros(num_completions, num_funcs, dtype=torch.float32)
        for i, (func_name, reward_func) in enumerate(zip(self.reward_func_names, self.reward_funcs)):
            origin_func_reward = reward_func(completions=completions, solution=solutions)

            if self.normalize_rule_reward:
                self.reward_normalizer.update(origin_func_reward)
                normalized_func_reward = self.reward_normalizer.norm(origin_func_reward)
                normalized_reward_tensor = torch.tensor(normalized_func_reward, dtype=torch.float32)
                rewards_per_func[:, i] = normalized_reward_tensor

            origin_reward_tensor = torch.tensor(origin_func_reward, dtype=torch.float32)
            rewards_per_func[:, i] = origin_reward_tensor
            reward_func_metrics[func_name] = origin_reward_tensor

        # 计算最终奖励值
        rewards = (rewards_per_func * self.reward_weights).sum(dim=1)
        if self.with_reward_baseline:
            self.reward_moving_avg.add(rewards)
            rewards -= torch.tensor(self.reward_moving_avg.get(), dtype=torch.float32)
        # 添加总奖励到指标中
        reward_func_metrics["total_reward"] = rewards

        return rewards, reward_func_metrics

    def aggregate_reward_metrics(self, experiences: List[Experience]) -> dict:
        """
        Aggregates reward function metrics across all experiences by reward function name.

        Args:
            experiences: List of Experience objects containing reward metrics

        Returns:
            dict: Mapping of reward function name to its average value across all experiences
        """
        # Initialize aggregated metrics dict
        aggregated_metrics = {}

        # Early return if no experiences or no metrics
        if not experiences or "reward_func_metrics" not in experiences[0].info:
            return aggregated_metrics

        # Get reward function names from first experience
        reward_names = experiences[0].info["reward_func_metrics"].keys()

        # Initialize lists to store values for each reward function
        for name in reward_names:
            aggregated_metrics[name] = []

        # Collect all values for each reward function
        for exp in experiences:
            metrics = exp.info["reward_func_metrics"]
            for name in reward_names:
                # Concatenate all tensor values for this reward function
                aggregated_metrics[name].extend(metrics[name])

        # Calculate mean for each reward function
        for name in reward_names:
            values = torch.cat(aggregated_metrics[name])
            aggregated_metrics[name] = values.mean().item()

        return aggregated_metrics

    @torch.no_grad()
    def compute_grpo_advantages(self, rewards):
        # Compute monte carlo trajectory-level advantage
        args = self.strategy.args
        rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
        values = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - values
        # Additionally normalize by std.
        std_grouped_rewards = rewards.std(dim=1, keepdim=True)
        advantages = advantages / (std_grouped_rewards + 1e-9)
        return advantages

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(self, batch_inputs, **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(batch_inputs, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, batch_inputs, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(batch_inputs, **generate_kwargs)

        # vLLM generation
        all_prompts = batch_inputs["prompt"]
        all_labels = batch_inputs["label"]
        samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)
        return samples

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        # 如果有初始模型，远程调用其 forward，异步获取 logprobs。
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, logps_allgather=True, packed_seq_lens=packed_seq_lens
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        # 如果有 critic，远程调用其 forward，异步获取 value。
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        # Rewards handling
        remote_rewards = []
        if self.reward_funcs:
            # 本地 reward 计算
            # Handle custom reward functions
            if not self.packing_samples:
                completion_ids = sequences[:, -num_actions:]
                completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            else:
                # For packed samples, we need to extract each completion separately
                completion_ids = []
                prompt_offset = 0
                tokens_list = sequences_cpu.tolist()[0]
                # num_actions 在 packed 模式下是一个列表，记录每个样本的 completion 长度
                for i, (seq_len, completion_len) in enumerate(zip(packed_seq_lens, num_actions)):
                    # prompt_len = seq_len - completion_len
                    completion_start = prompt_offset + (seq_len - completion_len)
                    completion_end = prompt_offset + seq_len
                    # 只取 completion 部分
                    completion_ids.append(tokens_list[completion_start:completion_end])
                    # 更新下一个样本的起始位置
                    prompt_offset += seq_len

                completion_ids = torch.tensor(completion_ids, device=sequences_cpu.device)
                completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            # Compute rewards using reward functions
            rewards, reward_func_metrics = self.compute_func_rewards(completions, samples.labels)

        # support remote RM API with ray
        elif self.remote_rm_url is not None:
            # remote RM
            if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

                if self.custom_reward_func:
                    rewards = self.custom_reward_func.remote(queries, samples.prompts, samples.labels)
                    remote_rewards.append(rewards)
                else:
                    for rm in self.remote_rm_url:
                        rewards = remote_rm_fn_ray.remote(
                            rm, queries=queries, prompts=samples.prompts, labels=samples.labels
                        )
                        remote_rewards.append(rewards)
            else:
                remote_rewards.append(ray.put(None))

        else:
            for rm in self.reward_model:
                remote_rewards.append(
                    rm.forward.remote(
                        sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens, pad_sequence=True
                    )
                )
        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(remote_rewards)
            ray.get([self.reward_model[0].empty_cache.remote()])

        # log probs
        # 本地 actor 计算 logprobs。
        action_log_probs = self.actor(
            sequences,
            num_actions,
            attention_mask,
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()

        # 如果 reward_funcs，获取 logprobs 和 value，reward 本地 to(device)。
        if self.reward_funcs:
            base_action_log_probs, value = ray.get([base_action_log_probs_ref, value_ref])
            base_action_log_probs = base_action_log_probs.to(device) if base_action_log_probs is not None else None
            value = value.to(device) if value is not None else None
            rewards = rewards.to(device)

        else:
            ref_values = ray.get([base_action_log_probs_ref, value_ref] + remote_rewards)
            base_action_log_probs, value = ref_values[0], ref_values[1]
            base_action_log_probs = base_action_log_probs.to(device) if base_action_log_probs is not None else None
            value = value.to(device) if value is not None else None
            rewards = [r.to(device) for r in ref_values[2:]]
            rewards = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        wait_time = time.time() - start

        # broadcast rewards to all ring attention ranks when using remote RM
        # 如果是远程 reward 或自定义 reward，且有 ring attention group，则主 rank 广播 reward 给其他 rank。
        if (self.remote_rm_url or self.reward_funcs) and self.strategy.ring_attn_group is not None:
            if self.strategy.ring_attn_rank == 0:
                dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                dist.broadcast_object_list(
                    rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )

        # avoid CUDA OOM when colocate models
        # 如果共置，清理 reward_model 显存。
        # 如果 actor 共置，同步并清理显存。
        if args.colocate_critic_reward and not self.remote_rm_url and not self.reward_funcs:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # 如果有初始模型且不用 KL loss，计算 KL 散度，否则全零。
        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        # 如果是打包样本，需 unpad/unpack，便于后续处理。
        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            if self.strategy.ring_attn_group is not None:
                assert samples.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=samples.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=value,
                    kl=kl,
                )
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        # 如果不用 KL loss，清空 base_action_log_probs。
        if not args.use_kl_loss:
            base_action_log_probs = None

        # 构造 info 字典，包含 KL、reward、长度等信息。
        # 如果有 reward_funcs，补充 reward_func_metrics。
        info = {
            "kl": kl_mean,
            "reward": rewards,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.reward_funcs:
            info.update(reward_func_metrics)

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        ray.get(refs)

        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        labels=labels,
                        pad_len=None,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                # pad seq makes the sequence a multiple of ring_attention_size.
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        labels=labels,
                        pad_len=pad_len,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
