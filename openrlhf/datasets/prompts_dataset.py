from torch.utils.data import Dataset
from tqdm import tqdm
from trl.data_utils import maybe_apply_chat_template


def preprocess_data(
    data,
    input_key="input",
    label_key=None,
    system_prompt=None,
    input_template=None,
    tokenizer=None,
    apply_chat_template: bool = False,
) -> str:
    keys = [key for key in data if key not in [input_key, label_key]]
    reward_kwargs = {key: data[key] for key in keys}
    if apply_chat_template:
        prompt = []
        if system_prompt is not None:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": data[input_key]})
        example = {"prompt": prompt}
        prompt_text = maybe_apply_chat_template(example, tokenizer)["prompt"]
    else:
        prompt_text = data[input_key]
        if input_template:
            prompt_text = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label_text = "" if label_key is None else data[label_key]
    processed_input = {"prompt": prompt_text, "label": label_text, **reward_kwargs}
    return processed_input


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        system_template=None,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        self.processed_inputs = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            processed_input = preprocess_data(
                data,
                input_key,
                label_key,
                system_template,
                input_template,
                tokenizer,
                apply_chat_template,
            )
            self.processed_inputs.append(processed_input)

    def __len__(self):
        length = len(self.processed_inputs)
        return length

    def __getitem__(self, idx):
        return self.processed_inputs[idx]
