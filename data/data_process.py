import datasets
import transformers
from threading import Lock
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


def load_and_templated_qa_data(dataset_name: str, split: str = 'train', q_col: str = 'Question',
                               a_col: str = 'Answer') -> datasets.Dataset:
    """Load dataset and apply a conversation template to each sample."""
    dataset = datasets.load_dataset(dataset_name, split=split)

    def apply_template(sample):
        sample['conversation'] = [{"role": "system", "content": "You are a helpful assistant."},
                                  {"role": "user", "content": sample[q_col]},
                                  {"role": "assistant", "content": sample[a_col]}]
        return sample

    dataset = dataset.map(apply_template, batched=False)
    return dataset


def split_dataset(dataset: datasets.Dataset, train_size: float = 0.8, seed: int = 42,
                  stratify_column: str = 'qtype') -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Split dataset into training and validation sets."""
    split_datasets = dataset.train_test_split(train_size=train_size, seed=seed, stratify_by_column=stratify_column)
    return split_datasets['train'], split_datasets['test']


class DataProcessor:
    __lock = Lock()
    __instance = None

    def __new__(cls, tokenizer_name: str):
        with cls.__lock:
            if cls.__instance is None:
                cls.__instance = super(DataProcessor, cls).__new__(cls)
                cls.__instance.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        return cls.__instance

    def __init__(self, tokenizer_name: str):
        pass  # Initialization is handled in __new__

    def tokenize(self, dataset: datasets.Dataset, text_column: str, batch_size: int = 32) -> DataLoader:
        def tokenize_function(examples):
            return self.tokenizer(examples[text_column], truncation=True, padding='max_length')

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        return DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)
