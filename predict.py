import itertools
from io import TextIOWrapper, StringIO
from typing import Union

import pandas as pd
import rich.progress
import torch
from rich import inspect, print
from torch.utils.data import DataLoader

import preprocess
import tune
from models import MultipleChoiceModelOutput
from params import PARAMS
from utils.stub import track
from utils.torchtools import get_device_of_module, on_device
from utils.typing import *


def predict_one(model: Model):
    device = get_device_of_module(model)

    def inner(batch: dict[str, torch.Tensor]):
        output: MultipleChoiceModelOutput = model(
            **{k: v.to(device) for k, v in batch.items()}
        )
        result: list[int] = output.logits.argmax(-1).tolist()
        return result

    return inner


def predict(
    model: Model,
    *,
    buffer: Union[TextIOWrapper, StringIO],
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
):
    with (
        torch.no_grad(),
        on_device(model, device) as model,
        PARAMS.progress(lambda: rich.progress.Progress()) as progress,
    ):
        model.eval()
        count_batch = len(dataloader)

        task = progress.add_task("Predicting...", total=count_batch)
        out = itertools.chain.from_iterable(
            map(
                predict_one(model),
                track(
                    hook_post=lambda idx, item: progress.update(
                        task,
                        description=f"Predict {idx+1} of {count_batch}.",
                        advance=1,
                    ),
                )(dataloader),
            )
        )
        # columns=["pos_idx"],

        buffer.write("[")
        for idx, each in enumerate(out):
            if idx != 0:
                buffer.write(",")
            buffer.write(f'{{"pos_idx":{each}}}')
        buffer.write("]")

    return model
