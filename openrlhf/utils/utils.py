import os
import math
import torch.nn as nn
import torch

from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def setup_tokenizer_and_resize(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrain, trust_remote_code=True, local_files_only=True, use_fast=use_fast
    )
    tokenizer.padding_side = padding_side

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Resize embeddings safely
    safe_resize_token_embeddings(model, len(tokenizer))

    return tokenizer


def safe_resize_token_embeddings(model, new_vocab_size):
    """
    自动判断是否为 NormHead 并安全地扩展 vocab，支持权重共享。
    """
    # 检查是否有权重共享
    shared_weights = (
        hasattr(model, "lm_head")
        and hasattr(model, "get_input_embeddings")
        and model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    )

    if hasattr(model, "resize_token_embeddings"):
        try:
            model.resize_token_embeddings(new_vocab_size)
            return  # 如果能正常处理，直接用
        except TypeError as e:
            print(f"[Warning] Default resize failed: {e}")
            print("-> Fallback to custom resize for non-Linear lm_head...")

    # 自定义处理
    resize_embeddings(model, new_vocab_size)

    if shared_weights:
        # 再次设置 lm_head.weight 与 embedding.weight 共享
        model.lm_head.weight = model.get_input_embeddings().weight
        print("-> Re-tied lm_head.weight with input embeddings.")
    else:
        custom_resize_lm_head(model, new_vocab_size)


def resize_embeddings(model, new_vocab_size):
    """
    Resize input embeddings manually if needed.
    """
    old_embeddings = model.get_input_embeddings()
    old_vocab_size, hidden_size = old_embeddings.weight.shape

    if new_vocab_size == old_vocab_size:
        return

    new_embed = nn.Embedding(new_vocab_size, hidden_size)
    new_embed.weight.data.normal_(mean=0.0, std=0.02)

    # Copy overlapping part
    num_tokens_to_copy = min(old_vocab_size, new_vocab_size)
    new_embed.weight.data[:num_tokens_to_copy] = old_embeddings.weight.data[:num_tokens_to_copy]

    model.set_input_embeddings(new_embed)
    print("-> Resized input embeddings.")


def custom_resize_lm_head(model, new_vocab_size):
    """
    处理自定义的 NormHead 类型。
    """
    lm_head = model.lm_head
    if hasattr(lm_head, "weight"):
        old_vocab_size, hidden_size = lm_head.weight.shape

        if new_vocab_size == old_vocab_size:
            return

        new_weight = nn.Parameter(torch.empty((new_vocab_size, hidden_size)))
        new_weight.data.normal_(mean=0.0, std=0.02)

        # Copy overlapping part safely
        num_tokens_to_copy = min(old_vocab_size, new_vocab_size)
        new_weight.data[:num_tokens_to_copy] = lm_head.weight.data[:num_tokens_to_copy]

        lm_head.weight = new_weight

        print(f"-> Resized lm_head from {old_vocab_size} to {new_vocab_size}.")
    else:
        raise ValueError("lm_head does not have a weight attribute.")


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    dataset_config=None,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv", ".parquet"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                strategy.print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset,name=dataset_config, data_dir=data_dir)
                strategy.print(f"loaded {dataset} from files")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, name=dataset_config, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")
