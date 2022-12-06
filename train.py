from typing import cast

import rich.progress
import torch
import torch.nn
import torch.optim
from torch.utils.data.dataloader import DataLoader
from transformers import (AutoModelForMultipleChoice, AutoTokenizer,
                          get_scheduler)

from params import PARAMS
from utils.typing import *


def get_pretrained() -> tuple[Tokenizer, Model]:
    """Get an `AutoTokenzier` and a `AutoModelForMultipleChoice`"""
    tokenizer = AutoTokenizer.from_pretrained(PARAMS.MODEL_PRETRAINED)
    model = AutoModelForMultipleChoice.from_pretrained(PARAMS.MODEL_PRETRAINED)
    return tokenizer, model

def train(
    model: Model,
    *,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Model:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 3
    num_training_steps = num_epochs * len(
        dataloader
    )
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    # accelerator = Accelerator()

    # dataloader, model, optimizer = typing_identity(accelerator.prepare)(dataloader, model, optimizer)

    model = model.to(device)

    for param in model.bert.encoder.layer[:9].parameters(): # type: ignore
        param.requires_grad = False # type: ignore
    
    model.train()
    batch_count = len(dataloader)
    epoch_count = 3

    with (
        rich.progress.Progress() as progress,
    ):
        epoch_task = progress.add_task("Epoch...", total=epoch_count)
        for idx_epoch in range(epoch_count):
            progress.update(epoch_task, description=f"Epoch {idx_epoch + 1} of {epoch_count}...")
            eval_task = progress.add_task("Evaluation...", total=batch_count)
            for idx_batch, batch in enumerate(dataloader):
                output = model(**{ k: v.to(device) for k, v in batch.items() })
                
                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()
                lr_scheduler.step()

                progress.update(eval_task, description=f"{idx_batch + 1} of {batch_count}. Loss {output.loss}")
                # if idx_batch % 4 == 0:
                #     print(output.logits.argmax(-1).cpu() - batch["labels"])
                progress.advance(eval_task)
            progress.advance(epoch_task)
        progress.refresh()
    return model.to("cpu")

def eval(
    model: Model,
    *,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Model:
    # accelerator = Accelerator()

    # dataloader, model, optimizer = typing_identity(accelerator.prepare)(dataloader, model, optimizer)

    model = model.to(device)
    
    model.eval()
    batch_count = len(dataloader)

    count_true = 0
    total = 0
    with (
        torch.no_grad(),
        rich.progress.Progress() as progress
    ):
        eval_task = progress.add_task("Evaluation...", total=batch_count)
        for idx_batch, batch in enumerate(dataloader):

            output = model(**{ k: v.to(device) for k, v in batch.items() if k != "labels" })
            count_true += (len(batch["labels"]) - (cast(torch.Tensor, output.logits).argmax(-1).cpu() - batch["labels"]).count_nonzero()).item()
            total += batch["labels"].shape[0]

            progress.update(eval_task, advance=1, description=f"{idx_batch + 1} of {batch_count}. Accuracy {count_true} of {total} ({count_true / total * 100 if total else 0.0:.2f}%)")

    # return accelerator.unwrap_model(model)
    progress.refresh()
    return model.to("cpu")

__all__ = ["get_pretrained", "train"]