import json
import re
from typing import Dict, List

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
    def __init__(self, **kwargs):
        self.config = kwargs
        
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        raise NotImplementedError
        
    def validate_input(self, completions, solution=None):
        """统一的输入验证"""
        pass



class MathAccuracyReward(BaseRewardFunction):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol,
                                extraction_mode='first_match',
                                extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
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
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
                print('Failed to parse gold solution: ', sol)
            rewards.append(reward)

        return rewards


class FormatReward(BaseRewardFunction):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific
        format."""
        pattern = r'^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$'

        matches = [
            re.match(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completions
        ]
        return [1.0 if match else 0.0 for match in matches]


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
        pattern = r'(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)'
        matches = [
            len(re.findall(pattern, content)) for content in completions
        ]
        # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
        return [min(1.0, count / 3) for count in matches]


class LengthReward(BaseRewardFunction):
    """Compute length-based rewards to discourage overthinking and promote
    token efficiency.
    # 基于Kimi 1.5技术报告
    # 考虑正确性和长度的平衡
    # 有最小/最大长度计算

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """

    def __call__(self, completions: list[Dict[str, str]], solution: list[str],
                 **kwargs) -> List[float]:
        # First check correctness of answers
        correctness = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(
                sol,
                extraction_mode='first_match',
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                # Skip unparseable examples
                correctness.append(
                    True)  # Treat as correct to avoid penalizing
                print('Failed to parse gold solution: ', sol)
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

        # Calculate lengths
        lengths = [len(content) for content in completions]
        min_len = min(lengths)
        max_len = max(lengths)

        # If all responses have the same length, return zero rewards
        if max_len == min_len:
            return [0.0] * len(completions)

        rewards = []
        for length, is_correct in zip(lengths, correctness):
            lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

            if is_correct:
                reward = lambda_val
            else:
                reward = min(0, lambda_val)

            rewards.append(float(reward))

        return rewards


class CosineScaledReward(BaseRewardFunction):
    """Reward function that scales based on completion length using a
    cosine schedule.
    # https://arxiv.org/abs/2502.03373

    Shorter correct solutions are rewarded more than longer ones.
    Longer incorrect solutions are penalized less than shorter ones.

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    This function is parameterized by the following arguments:
        min_value_wrong: Minimum reward for wrong answers
        max_value_wrong: Maximum reward for wrong answers
        min_value_correct: Minimum reward for correct answers
        max_value_correct: Maximum reward for correct answers
        max_len: Maximum length for scaling
    """
    def __init__(
        self,
        cosine_min_len_value_wrong: float = -1.0,
        cosine_max_len_value_wrong: float = -0.5,
        cosine_min_len_value_correct: float = 0.5,
        cosine_max_len_value_correct: float = 1.0,
        cosine_max_len: int = 1000,
        accuracy_orm=None,
    ):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracyReward()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        # Apply cosine scaling based on length

        import math
        cosin = math.cos(t * math.pi / T)
        return min_value +  0.5 *(max_value -
                            min_value) * (1.0 + cosin)

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.0
            if is_correct:
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(content)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
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
