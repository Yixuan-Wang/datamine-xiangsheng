from typing import Literal

import pandas as pd
from datasets import Dataset  # type: ignore
from torch.utils.data.dataloader import DataLoader

import utils.collate
from params import PARAMS
from utils.typing import *

__all__ = []

def get_qa_pair(tokenizer: Tokenizer):
    def inner(row):
        question, choices, answer = row["src"], row["choices"], row["pos_idx"]
        question = question.split("|")[-1]
        return tokenizer([question] * len(choices), choices) | {"labels": answer}

    return inner


def get_dataloader(
    dataset_name: "Literal['train'] | Literal['valid'] | Literal['test']" = "train",
    *,
    tokenizer: Tokenizer
):
    df: pd.DataFrame = (
        pd.read_json(f"data/{dataset_name}.json").join(
            pd.read_json(f"data/{dataset_name}_label.json")
        )
    )#[
    #    :1000
    #] # TODO Debug, remove this

    if dataset_name == "train":
        df = df[:50000]
    elif dataset_name == "valid":
        df = df[:10000]

    dataset = Dataset.from_pandas(pd.DataFrame.from_records(
        df.apply(get_qa_pair(tokenizer), axis=1) # type: ignore
    ))
    dataset.set_format("torch")

    shuffle: bool = True

    return DataLoader(
        dataset=dataset,  # type: ignore
        shuffle=shuffle,
        batch_size=PARAMS.BATCH_SIZE,
        collate_fn=utils.collate.data_collator,
    )

__all__.append("get_dataloader")
