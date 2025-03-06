import json
import math
import re
from typing import Dict, List, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from transformers.utils.import_utils import _is_package_available

# Use same as transformers.utils.import_utils
_e2b_available = _is_package_available('e2b')


def is_e2b_available() -> bool:
    return _e2b_available


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import Sandbox

    load_dotenv()


class BaseRewardFunction:
    """Placeholder for the base reward function class."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        raise NotImplementedError

    def validate_input(self, completions, solution=None):
        """统一的输入验证."""
        pass


class MathAccuracyReward(BaseRewardFunction):
    """Reward function to check if the model's response is mathematically
    equivalent to the ground truth solution. Uses latex2sympy2 for parsing and
    math_verify for validation.

    If the model answer is mathematically correct, we assign a reward of 1.0.
    If it is incorrect, the reward is 0.0. In cases where the ground truth
    solution cannot be parsed,     we assign a neutral reward of 0.5 to avoid
    unfair penalties.
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            # Parse the ground truth solution
            gold_parsed = parse(sol,
                                extraction_mode='first_match',
                                extraction_config=[LatexExtractionConfig()])
            if gold_parsed:
                # Check if parsing was successful
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,  # 不允许畸形运算符
                                basic_latex=True,  # 使用基本LaTeX
                                equations=True,  # 允许方程
                                boxed='all',  # 所有boxed内容
                                units=True,  # 允许单位
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,  # boxed匹配优先级
                            try_extract_without_anchor=False,  # 不尝试无锚点提取
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                try:
                    reward = float(verify(answer_parsed, gold_parsed))
                except Exception as e:
                    print(
                        f'verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}'
                    )
                    reward = 0.0
            else:
                # 标准答案解析失败时给予中性奖励
                # If the gold solution is not parseable, we reward 0.5 to skip this example
                reward = 0.5
                print('Failed to parse gold solution: ', sol)
            rewards.append(reward)

        return rewards


class FormatReward(BaseRewardFunction):
    """Reward function to check if the completion follows the correct format:

    Expected format:
    ```
    <think>...</think> <answer>...</answer>
    ```

    - Matches two possible formats:
        1. Multi-line format:
            ```
            <think>
            ...
            </think>
            <answer>
            ...
            </answer>
            ```
        2. Single-line format:
            ```
            <think>...</think> <answer>...</answer>
            ```

    A reward of `1.0` is given for correctly formatted responses, otherwise `0.0`.
    """

    def __init__(self):
        # Regular expression patterns for checking format validity
        self.patterns = [
            re.compile(r'^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$',
                       re.DOTALL | re.MULTILINE),
            re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])',
                       re.DOTALL | re.MULTILINE),
        ]

    def is_valid_format(self, content: str) -> bool:
        """Checks if the given content matches at least one of the valid
        formats.

        Args:
            content (str): The completion text to validate.

        Returns:
            bool: True if the format is valid, False otherwise.
        """
        return any(pattern.match(content) for pattern in self.patterns)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Evaluates whether each completion follows the expected format.

        Args:
            completions (List[str]): List of generated completions.

        Returns:
            List[float]: List of rewards (1.0 if formatted correctly, else 0.0).
        """
        return [
            1.0 if self.is_valid_format(content) else 0.0
            for content in completions
        ]


class ReActFormat(BaseRewardFunction):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific
        format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [
            re.match(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completions
        ]
        return [1.0 if match else 0.0 for match in matches]


class TagCountReward(BaseRewardFunction):
    """Reward function that checks if we produce the desired number of think
    and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count('<think>\n') == 1:
            count += 0.25
        if text.count('\n</think>\n') == 1:
            count += 0.25
        if text.count('\n<answer>\n') == 1:
            count += 0.25
        if text.count('\n</answer>') == 1:
            count += 0.25

        return count

    def __call__(self, completions, **kwargs) -> List[float]:

        return [self.count_tags(c) for c in completions]


class ReasoningStepReward(BaseRewardFunction):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        # Regex pattern to find indicators of reasoning steps
        pattern = r'(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)'
        # Count the number of reasoning step indicators in each completion
        matches = [
            len(re.findall(pattern, content)) for content in completions
        ]
        # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
        return [min(1.0, count / 3) for count in matches]


class LengthReward(BaseRewardFunction):
    """Computes length-based rewards to discourage overthinking and promote
    token efficiency.

    Reference: Kimi 1.5 tech report (https://arxiv.org/abs/2501.12599)

    This function balances correctness and response length, rewarding concise correct answers.

    Reward Calculation:
    - Correct answers: `reward = 0.5 - (length - min_length) / (max_length - min_length)`
    - Incorrect answers: `reward = min(0, 0.5 - (length - min_length) / (max_length - min_length))`

    Args:
        completions (List[Dict[str, str]]): List of model-generated completions.
        solution (List[str]): List of ground truth solutions.

    Returns:
        List[float]: List of computed rewards for each completion.
    """

    def check_correctness(self, completions: List[Dict[str, str]],
                          solutions: List[str]) -> List[bool]:
        """Checks the correctness of each completion by comparing it to the
        ground truth solution.

        Args:
            completions (List[Dict[str, str]]): Generated completions.
            solutions (List[str]): Ground truth solutions.

        Returns:
            List[bool]: A list indicating whether each completion is correct.
        """
        correctness = []
        for content, sol in zip(completions, solutions):
            gold_parsed = parse(
                sol,
                extraction_mode='first_match',
                extraction_config=[LatexExtractionConfig()],
            )

            if not gold_parsed:
                # Treat as correct to avoid penalization when parsing fails
                correctness.append(True)
                print(f'Failed to parse gold solution: {sol}')
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode='first_match',
            )

            correctness.append(verify(answer_parsed, gold_parsed))

        return correctness

    def __call__(self, completions: List[Dict[str, str]], solution: List[str],
                 **kwargs) -> List[float]:
        """Computes length-based rewards for each completion.

        Args:
            completions (List[Dict[str, str]]): List of model-generated completions.
            solution (List[str]): List of ground truth solutions.

        Returns:
            List[float]: List of computed rewards.
        """
        correctness = self.check_correctness(completions, solution)

        # Calculate response lengths
        lengths = [len(content) for content in completions]
        min_len, max_len = min(lengths), max(lengths)

        # If all responses are of equal length, return zero rewards
        if max_len == min_len:
            return [0.0] * len(completions)

        rewards = []
        for length, is_correct in zip(lengths, correctness):
            lambda_val = 0.5 - (length - min_len) / (max_len - min_len)
            reward = lambda_val if is_correct else min(0, lambda_val)
            rewards.append(float(reward))

        return rewards


class CosineScaledReward(BaseRewardFunction):
    """Reward function that scales based on completion length using a cosine
    schedule.

    Reference: https://arxiv.org/abs/2502.03373

    Shorter correct completions receive higher rewards.
    Longer incorrect completions receive lower penalties.

    Args:
        cosine_min_value_wrong (float): Minimum reward for incorrect answers.
        cosine_max_value_wrong (float): Maximum reward for incorrect answers.
        cosine_min_value_correct (float): Minimum reward for correct answers.
        cosine_max_value_correct (float): Maximum reward for correct answers.
        cosine_max_len (int): Maximum length for scaling.
        accuracy_orm (BaseRewardFunction, optional): Accuracy computation module.

    Example:
        >>> reward_fn = CosineScaledReward()
        >>> rewards = reward_fn(["答案是42", "错误答案"], ["答案是42", "答案是43"])
        >>> print(rewards)
    """

    def __init__(
        self,
        cosine_min_value_wrong: float = -0.5,
        cosine_max_value_wrong: float = -1.0,
        cosine_min_value_correct: float = 0.5,
        cosine_max_value_correct: float = 1.0,
        cosine_max_len: int = 1000,
        accuracy_orm: Union[BaseRewardFunction, None] = None,
    ):
        self.min_value_wrong = cosine_min_value_wrong
        self.max_value_wrong = cosine_max_value_wrong
        self.min_value_correct = cosine_min_value_correct
        self.max_value_correct = cosine_max_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracyReward()

    @staticmethod
    def cosine_scaled_reward(t: int, T: int, min_value: float,
                             max_value: float) -> float:
        """Computes a cosine-scaled reward value based on response length.

        Args:
            t (int): Current length of the completion.
            T (int): Maximum length for scaling.
            min_value (float): Minimum reward value.
            max_value (float): Maximum reward value.

        Returns:
            float: Scaled reward value.
        """
        cosine_value = math.cos(t * math.pi / T)
        return min_value + 0.5 * (max_value - min_value) * (1.0 + cosine_value)

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs) -> List[float]:
        """Computes cosine-scaled rewards for a list of model completions.

        Args:
            completions (List[str]): List of generated completions.
            solution (List[str]): List of ground truth solutions.
            **kwargs: Additional arguments for the accuracy function.

        Returns:
            List[float]: List of computed rewards.
        """
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []

        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.0
            min_value = self.min_value_correct if is_correct else self.max_value_wrong
            max_value = self.max_value_correct if is_correct else self.min_value_wrong
            gen_text_len = len(content)
            reward = cosine_scaled_reward(gen_text_len, self.max_len,
                                          min_value, max_value)
            rewards.append(reward)

        return rewards


class RepetitionPenalty(BaseRewardFunction):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 repetition_n_grams: int = 3,
                 repetition_max_penalty: float = -1.0):
        """Computes N-gram repetition penalty as described in Appendix C.2 of
        https://arxiv.org/abs/2502.03373. Reference implementation from:
        https://github.com/eddycmu/demystify-long-
        cot/blob/release/openrlhf/openrlhf/reward/repetition.py.

        Args:
        ngram_size: size of the n-grams
        max_penalty: Maximum (negative) penalty for wrong answers
        """
        if repetition_max_penalty > 0:
            raise ValueError(
                f'max_penalty {repetition_max_penalty} should not be positive')

        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """reward function the penalizes repetitions.

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


def extract_code(completion: str) -> str:
    pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ''
    return extracted_answer


class CodeReward(BaseRewardFunction):
    """Reward function that evaluates code snippets using the E2B code
    interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:

        if not is_e2b_available():
            raise ImportError(
                'E2B is not available and required for this reward function. Please install E2B with '
                '`pip install e2b-code-interpreter` and add an API key to a `.env` file.'
            )

        rewards = []
        # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
        try:
            """Returns a reward function that evaluates code snippets in a
            sandbox."""
            evaluation_script_template = """
            import subprocess
            import json

            def evaluate_code(code, test_cases):
                passed = 0
                total = len(test_cases)
                exec_timeout = 5

                for case in test_cases:
                    process = subprocess.run(
                        ["python3", "-c", code],
                        input=case["input"],
                        text=True,
                        capture_output=True,
                        timeout=exec_timeout
                    )

                    if process.returncode != 0:  # Error in execution
                        continue

                    output = process.stdout.strip()
                    if output.strip() == case["output"].strip():
                        passed += 1

                success_rate = (passed / total)
                return success_rate

            code_snippet = {code}
            test_cases = json.loads({test_cases})

            evaluate_code(code_snippet, test_cases)
            """
            verification_info = kwargs['verification_info']
            scripts = [
                evaluation_script_template.format(
                    code=json.dumps(code),
                    test_cases=json.dumps(json.dumps(info['test_cases'])))
                for code, info in zip(completions, verification_info)
            ]
            with Sandbox(timeout=30, request_timeout=3) as sbx:
                for script in scripts:
                    execution = sbx.run_code(
                        script, language=verification_info['language'])
                    try:
                        output = float(execution.text)
                    except (TypeError, ValueError):
                        output = 0.0
                    rewards.append(output)
        except Exception as e:
            print(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


relu_based_reward_func_mapping = {
    'accuracy': MathAccuracyReward,
    'format': FormatReward,
    'react_format': ReActFormat,
    'tag_reward': TagCountReward,
    'reasoning_steps': ReasoningStepReward,
    'length': LengthReward,
    'cosine': CosineScaledReward,
    'repetition': RepetitionPenalty,
}


def cosine_scaled_reward(t: int, T: int, min_value: float,
                         max_value: float) -> float:
    """Computes a cosine-scaled reward value based on length.

    :param t: Current length of the response.
    :param T: Maximum length considered.
    :param min_value: Minimum reward value.
    :param max_value: Maximum reward value.
    :return: Scaled reward value.
    """
    cosine_value = math.cos(t * math.pi / T)
    return min_value + 0.5 * (max_value - min_value) * (1.0 + cosine_value)


def test_cosine_scaled_reward_behavior() -> None:
    """Tests the cosine_scaled_reward function for correct and incorrect
    answers with varying lengths, ensuring expected reward behavior."""
    # Test cases: correct answers (varying lengths)
    correct_answers = [
        '答案是42',  # Length: 5
        '经过仔细计算，答案是42',  # Length: 12
        '让我详细解释一下，经过认真思考和计算，最终答案是42'  # Length: 25
    ]

    # Test cases: incorrect answers (varying lengths)
    wrong_answers = [
        '答案是24',  # Length: 5
        '经过仔细计算，答案是24',  # Length: 12
        '让我详细解释一下，经过认真思考和计算，最终答案是24'  # Length: 25
    ]

    # Ground truth correctness (1 = correct, 0 = incorrect)
    accuracy_rewards: List[int] = [1, 1, 1, 0, 0, 0]
    all_contents = correct_answers + wrong_answers

    # Reward scaling parameters
    min_value_wrong, max_value_wrong = -1.0, -0.5
    min_value_correct, max_value_correct = 0.5, 1.0
    max_len = 20

    # Compute cosine-scaled rewards
    cosine_rewards = []
    for content, acc_reward in zip(all_contents, accuracy_rewards):
        gen_len = len(content)
        is_correct = acc_reward >= 1.0

        if is_correct:
            min_value = min_value_correct
            max_value = max_value_correct
        else:
            # Swap min/max for incorrect answers
            min_value = max_value_wrong
            max_value = min_value_wrong

        reward = cosine_scaled_reward(gen_len, max_len, min_value, max_value)
        cosine_rewards.append(reward)

    # Assertions to verify expected reward behavior
    assert cosine_rewards[0] > cosine_rewards[1] > cosine_rewards[
        2], 'Correct answers should receive decreasing rewards as length increases.'
    assert cosine_rewards[3] < cosine_rewards[4] < cosine_rewards[
        5], 'Incorrect answers should receive decreasing penalties as length increases.'

    print('All tests passed successfully!')


if __name__ == '__main__':
    test_cosine_scaled_reward_behavior()
