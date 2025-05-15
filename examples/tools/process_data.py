import os
from datasets import load_dataset, load_from_disk


def load_data_from_disk_or_hf(data_name):
    if os.path.exists(data_name):
        dataset = load_from_disk(data_name)
    else:
        dataset = load_dataset(data_name)
    return dataset


def save_dataset_as_jsonl(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for split in dataset.keys():
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        dataset[split].to_json(output_path, orient="records", lines=True)
        print(f"Saved {split} split to {output_path}")


if __name__ == "__main__":
    # 示例用法
    data_name = "/root/llmtuner/llm/understand-r1-zero/datasets/train/math_lvl3to5_8k"
    output_dir = "/root/llmtuner/hfhub/datasets/math_lvl3to5_8k"

    # dataset = load_data_from_disk_or_hf(data_name)
    # save_dataset_as_jsonl(dataset, output_dir)

    dataset = load_dataset(
        "json",
        data_files="/root/llmtuner/hfhub/datasets/math_lvl3to5_8k/train.jsonl",
        split="train",
    )

    print(dataset)
