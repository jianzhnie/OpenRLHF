def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


class CodeReward(BaseRewardFunction):
    """Reward function that evaluates code snippets using the E2B code
    interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        if not is_e2b_available():
            raise ImportError(
                "E2B is not available and required for this reward function. Please install E2B with "
                "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
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
            verification_info = kwargs["verification_info"]
            scripts = [
                evaluation_script_template.format(
                    code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"]))
                )
                for code, info in zip(completions, verification_info)
            ]
            with Sandbox(timeout=30, request_timeout=3) as sbx:
                for script in scripts:
                    execution = sbx.run_code(script, language=verification_info["language"])
                    try:
                        output = float(execution.text)
                    except (TypeError, ValueError):
                        output = 0.0
                    rewards.append(output)
        except Exception as e:
            logger.info(f"Error from E2B executor: {e}")
            rewards = [0.0] * len(completions)
        return rewards
