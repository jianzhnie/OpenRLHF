import json
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.data_utils import maybe_apply_chat_template

# === System prompts used for formatting conversations ===
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    "qwen_math_cot": "Please reason step by step, and put your final answer within \\boxed{}.",
    "deepseek_r1": (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it."
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    ),
    "open_r1": (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses."
        "You first think about the reasoning process as an internal monologue and then provide the user with the answer."
        "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    ),
    "no": None,
}

# === Optional template for formatting input ===
INPUT_TEMPLATE: Dict[str, str] = {
    "prompt": ("{instruction}. \n\n Please reason step by step, and put your final answer within \\boxed{{}}.")
}


def preprocess_data(
    data: Dict[str, Any],
    input_key: str = "input",
    label_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    input_template: Optional[Dict[str, str]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Dict[str, str]:
    """
    Preprocess a single data entry into a prompt-answer pair.

    Args:
        data: Dictionary representing one example from the dataset.
        input_key: Key used to retrieve the input prompt text.
        label_key: Key used to retrieve the label or answer.
        system_prompt: Optional system prompt to format the conversation.
        input_template: Optional formatting template for the input prompt.
        tokenizer: Tokenizer used with maybe_apply_chat_template.

    Returns:
        A dictionary with 'prompt' and 'answer' keys.
    """
    input_prompt: str = data.get(input_key, "")

    if system_prompt and tokenizer:
        # Use chat format
        prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}]
        example = {"prompt": prompt}
        prompt_text: str = maybe_apply_chat_template(example, tokenizer)["prompt"]
    elif input_template:
        # Use custom input template
        prompt_text = input_template["prompt"].format(instruction=input_prompt)
    else:
        prompt_text = input_prompt

    label_text = data.get(label_key, "") if label_key else ""

    return {
        "prompt": prompt_text,
        "answer": label_text,
    }


def process_and_save_dataset(
    data_path: str,
    model_name_or_path: str,
    output_path: str,
    input_key: str = "Problem",
    label_key: str = "Answer",
    system_prompt: Optional[str] = None,
    input_template: Optional[Dict[str, str]] = None,
) -> None:
    """
    Load a dataset, process each entry, and save as a line-separated JSONL file.

    Args:
        data_path: Path to the dataset (Hugging Face format or local).
        model_name_or_path: Model name or path to load tokenizer from.
        output_path: Output file path (.jsonl format).
        input_key: Dataset key for the input text.
        label_key: Dataset key for the answer/label.
        system_prompt: Optional system prompt for formatting.
        input_template: Optional template to format input text.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Load dataset
    dataset: Dataset = load_dataset(data_path, split="train")

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Process and save each example
    with output_file.open("w", encoding="utf-8") as f:
        for i, example in enumerate(dataset, start=1):
            processed_example = preprocess_data(
                example,
                input_key=input_key,
                label_key=label_key,
                system_prompt=system_prompt,
                input_template=input_template,
                tokenizer=tokenizer,
            )
            # Write each example as a single line JSON
            json_line = json.dumps(processed_example, ensure_ascii=False)
            f.write(json_line + "\n")
            if i % 10 == 0:
                print(f"Processed {i} examples")

    print(f"Finished processing {len(dataset)} examples")
    print(f"Saved processed data to {output_file}")


if __name__ == "__main__":
    # === Configuration ===
    data_path = "/root/llmtuner/hfhub/datasets/Maxwell-Jia/AIME_2024"
    model_name_or_path = "/root/llmtuner/hfhub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    output_path = "./data/aime24_qwen_math_cot.jsonl"

    # === Run Processing ===
    process_and_save_dataset(
        data_path=data_path,
        model_name_or_path=model_name_or_path,
        output_path=output_path,
        input_key="Problem",
        label_key="Answer",
        system_prompt=SYSTEM_PROMPT_FACTORY["qwen_math_cot"],
        input_template=None,  # or use INPUT_TEMPLATE if needed
    )
