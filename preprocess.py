from codecs import ignore_errors
from typing import Literal
import itertools

import rich.status
import pandas as pd
from datasets import Dataset  # type: ignore
from torch.utils.data.dataloader import DataLoader

import utils.collate
from params import PARAMS
from utils.typing import *
from utils.stub import window


def get_mcq_datum(tokenizer: Tokenizer):
    def inner(row: pd.Series):
        question, choices, answer = row["src"], row["choices"], row["pos_idx"]
        question_segment = "[SEP]".join(question.split("|")[-PARAMS.CONTEXT_LENGTH :])
        return tokenizer([question_segment] * len(choices), choices) | {
            "labels": answer
        }

    return inner


def get_mcq_test_datum(
    tokenizer: Tokenizer, context_length: int = PARAMS.CONTEXT_LENGTH
):
    def inner(row: pd.Series):
        question, choices = row["src"], row["choices"]
        question_segment = "[SEP]".join(question.split("|")[-context_length:])
        return tokenizer([question_segment] * len(choices), choices)

    return inner


def gen_nsp_true_datum(tokenizer: Tokenizer):
    def inner(row: pd.Series):
        question, choices, answer = row["src"], row["choices"], row["pos_idx"]
        # question_segment = "[SEP]".join(question.split("|")[-PARAMS.CONTEXT_LENGTH :])
        # return tokenizer(question_segment, choices[answer]) | {"labels": 0}
        yield from (
            tokenizer("[SEP]".join(sents[:-1]), sents[-1]) | {"labels": 0}
            for sents in window(
                itertools.chain(
                    question.split("|")[-PARAMS.CONTEXT_LENGTH - 2 :], choices[answer]
                ),
                PARAMS.CONTEXT_LENGTH + 1,
            )
        )

    return inner


def gen_nsp_false_datum(tokenizer: Tokenizer):
    def inner(row: pd.Series):
        question, choices, answer = row["src"], row["choices"], row["pos_idx"]
        question_segment = "[SEP]".join(question.split("|")[-PARAMS.CONTEXT_LENGTH :])
        yield from (
            tokenizer(question_segment, wrong_choice) | {"labels": 1}
            for idx, wrong_choice in enumerate(choices)
            if idx != answer
        )

    return inner


def get_dataloader(
    dataset_name: "Literal['train'] | Literal['valid']", *, tokenizer: Tokenizer
):
    df: pd.DataFrame = pd.read_json(f"data/{dataset_name}.json").join(
        pd.read_json(f"data/{dataset_name}_label.json")
    )

    with rich.status.Status("Sampling data") as status:
        if dataset_name == "train":
            df = df[: PARAMS.DATA_SIZE].reset_index(drop=True)  # type: ignore
            # status.status = "Sampling 1000 for [bold]train[/]"
            # df = df.sample(n=1000, ignore_index=True)
        elif dataset_name == "valid" and PARAMS.VALID_SIZE:
            status.status = f"Sampling {PARAMS.VALID_SIZE} for validation."
            df = df.sample(n=PARAMS.VALID_SIZE, ignore_index=True)

    dataset = Dataset.from_pandas(
        pd.DataFrame.from_records(
            df.apply(get_mcq_datum(tokenizer), axis=1)  # type: ignore
        )
    )
    dataset.set_format("torch")

    shuffle: bool = dataset_name == "train"

    return DataLoader(
        dataset=dataset,  # type: ignore
        shuffle=shuffle,
        batch_size=PARAMS.BATCH_SIZE,
        collate_fn=utils.collate.data_collator(),
    )


def get_test_dataloader(*, tokenizer: Tokenizer):
    df = pd.read_json(f"data/test.json")
    dataset = Dataset.from_pandas(
        pd.DataFrame.from_records(
            df.apply(get_mcq_test_datum(tokenizer), axis=1)  # type: ignore
        )
    )
    dataset.set_format("torch")

    return DataLoader(
        dataset=dataset,  # type: ignore
        shuffle=False,
        batch_size=PARAMS.BATCH_SIZE,
        collate_fn=utils.collate.data_collator(has_labels=False),
    )


def get_nsp_dataloader(*, tokenizer: Tokenizer):
    df: pd.DataFrame = pd.read_json(f"data/train.json").join(
        pd.read_json(f"data/train_label.json")
    )

    df = df.sample(n=5000, ignore_index=True)

    with rich.status.Status("Generating pretrain data") as status:
        dataset = Dataset.from_pandas(
            pd.DataFrame.from_records(
                itertools.chain(
                    itertools.chain.from_iterable(df.apply(gen_nsp_true_datum(tokenizer), axis=1)),  # type: ignore
                    itertools.chain.from_iterable(df.apply(gen_nsp_false_datum(tokenizer), axis=1)),  # type: ignore
                )
            )
        )

    dataset.set_format("torch")

    shuffle: bool = True

    return DataLoader(
        dataset=dataset,  # type: ignore
        shuffle=shuffle,
        batch_size=64,
        collate_fn=utils.collate.data_collator_for_nsp,
    )


__all__ = ["get_dataloader"]
