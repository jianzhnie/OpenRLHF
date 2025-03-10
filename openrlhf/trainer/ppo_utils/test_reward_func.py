from typing import List, Tuple

from openrlhf.utils.logging_utils import init_logger
from openrlhf.trainer.ppo_utils.reward_funcs import MathAccuracyReward, CosineScaledReward, MathAccuracyRewardV2


logger = init_logger(__name__)


def test_rewards_func_exam1() -> None:
    """Test the reward function with various math-related completion
    examples."""
    reward_fn = CosineScaledReward()
    # Test cases: pairs of (generated answer, expected solution)
    examples: List[Tuple[str, str]] = [
        ('so that x == 1 or x == 2, thus the result is $2*\pi*r$',
         '$2*\pi*r$'), ('The answer is $$\\sin(x)$$', '$$\\sin(x)$$'),
        ('After solving, we get $1/2$', '$1/2$'),
        ('The final result is $$(a + b)^2$$', '$(a + b)^2$'),
        ('Therefore, $$3!$$', '$3!$'),
        ('The point coordinates are $(1,2)$', '$(1,2)$')
    ]

    completions, solutions = zip(*examples)
    rewards = reward_fn(list(completions), list(solutions))

    logger.info('\nReward Examples:')
    for comp, solution, reward in zip(completions, solutions, rewards):
        logger.info(f'Answer: {comp}')
        logger.info(f'Gold: {solution}')
        logger.info(f'Length: {len(comp)}')
        logger.info(f'Reward: {reward:.3f}\n')


def test_rewards_func_exam2() -> None:
    """Test the reward function with different mathematical reasoning examples.

    Includes correct, incorrect, and partially correct responses.
    """
    examples = [
        # ✅ 正确示例（包含推理步骤）
        (r'首先，我们知道二次方程 $2x - 3 = 0$。\n'
         r'移项得到 $2x = 3$。\n'
         r'两边同时除以 2，得出 $x = \frac{3}{2}$。', r'x = \frac{3}{2}'),  # 完整推理 + 正确答案
        (r'根据爱因斯坦的质能方程：\n'
         r'$E = mc^2$。\n'
         r'其中，$m$ 代表质量，$c$ 代表光速。', r'E = mc^2'),  # 物理公式 + 解释
        (r'计算定积分 $\int_0^1 x^2 \,dx$。\n'
         r'首先，计算不定积分：$\int x^2 \,dx = \frac{x^3}{3}$。\n'
         r'然后代入上限 1 和下限 0，得到：\n'
         r'$\frac{1^3}{3} - \frac{0^3}{3} = \frac{1}{3}$。',
         r'\int_0^1 x^2 \,dx = \frac{1}{3}'),  # 清晰的积分推导

        # ❌ 错误示例（包含推理错误）
        (r'解方程 $2x - 3 = 0$。\n'
         r'移项得到 $2x = 3$。\n'
         r'然后两边同时除以 **3**，得出 $x = \frac{3}{3} = 1$。', r'x = \frac{3}{2}'
         ),  # 计算错误（除错数）
        (r'根据物理公式：$E = mc^3$。\n'
         r'但实际上光速的指数应为 2，因此正确公式是 $E = mc^2$。', r'E = mc^2'),  # 公式错误
        (r'计算定积分 $\int_0^1 x^2 \,dx$。\n'
         r'计算不定积分：$\int x^2 \,dx = \frac{x^3}{3}$。\n'
         r'然后代入上限 1 和下限 0，得到：\n'
         r'$\frac{1^3}{2} - \frac{0^3}{2} = \frac{1}{2}$。',
         r'\int_0^1 x^2 \,dx = \frac{1}{3}'),  # 积分结果错误

        # 🔄 部分正确（推理清晰但格式问题或等价表达）
        (r'解方程 $2x - 3 = 0$。\n'
         r'移项得到 $2x = 3$。\n'
         r'两边同时除以 2，得出 $x = 1.5$。', r'x = \frac{3}{2}'),  # 结果正确但不是 LaTeX 形式
        (r'级数求和结果如下：\n'
         r'$\frac{\pi^2}{6} = \sum_{n=1}^{\infty} \frac{1}{n^2}$。',
         r'\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}'),  # 数学等价但顺序不同
        (r'计算定积分 $\int_0^1 x^2 \,dx$。\n'
         r'计算不定积分：$\int x^2 \,dx = \frac{x^3}{3}$。\n'
         r'然后代入上下限，得到 $x^3/3$ 的变化量。\n'
         r'最终答案为 $\frac{2}{6}$。',
         r'\int_0^1 x^2 \,dx = \frac{1}{3}'),  # 结果正确但未化简
        (r'爱因斯坦公式 $E = c^2 m$ 适用于质量与能量的转换。', r'E = mc^2'),  # 变量顺序错误但等价
    ]

    reward_fn = CosineScaledReward(cosine_max_len=30)
    reward_fn = MathAccuracyReward()
    reward_fn = MathAccuracyRewardV2()

    completions, solutions = zip(*examples)
    rewards = reward_fn(list(completions), list(solutions))

    logger.info('\nReward Examples:')
    for comp, solution, reward in zip(completions, solutions, rewards):
        logger.info(f'Answer: {comp}')
        logger.info(f'Gold: {solution}')
        logger.info(f'Length: {len(comp)}')
        logger.info(f'Reward: {reward:.3f}\n')


if __name__ == '__main__':
    test_rewards_func_exam1()
    test_rewards_func_exam2()
