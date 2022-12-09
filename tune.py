from typing import Optional, cast, TypeVar
from pathlib import Path

import rich.progress
import rich
import torch
import torch.nn
import torch.optim
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModel, AutoTokenizer, BertModel, get_scheduler

import preprocess
from params import PARAMS
from utils.typing import *
from utils.torchtools import get_device_of_module, on_device
import models


def infer_model(name: str) -> Model:
    possible_models = list(Path("models").glob(f"model.{name}*.pt"))

    if len(possible_models) < 1:
        rich.print(f"[red bold]Cannot find model of name [underline]{name}[/].")
        raise ValueError
    elif len(possible_models) > 1:
        rich.print(f"[yellow bold]Ambiguous model name [underline]{name}[/].")
        rich.print([path.name for path in possible_models])
        raise ValueError

    model = torch.load(possible_models[0])
    return model


def get_pretrained(name: Optional[str] = None) -> tuple[Tokenizer, Model]:
    """Get an `AutoTokenzier` and a `AutoModelForMultipleChoice`"""
    model = (
        AutoModel.from_pretrained(PARAMS.MODEL_PRETRAINED)
        if name is None
        else infer_model(name)
    )
    tokenizer = AutoTokenizer.from_pretrained(PARAMS.MODEL_PRETRAINED)
    return tokenizer, model


def train(
    model: models.ModelMultipleChoice,
    *,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    dataloader_eval: Optional[DataLoader] = None,
) -> models.ModelMultipleChoice:
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMS.LEARNING_RATE)

    count_epoch = 3
    num_training_steps = count_epoch * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    # accelerator = Accelerator()

    # dataloader, model, optimizer = typing_identity(accelerator.prepare)(dataloader, model, optimizer)
    device_prev = get_device_of_module(model)
    model = model.to(device)

    for param in model.bert.encoder.layer[: PARAMS.FREEZE_LAYERS].parameters():  # type: ignore
        param.requires_grad = False  # type: ignore

    count_batch = len(dataloader)
    count_epoch = 3

    with (PARAMS.progress(lambda: rich.progress.Progress()) as progress,):
        task_epoch = progress.add_task("Training...", total=count_epoch)
        for idx_epoch in range(count_epoch):
            progress.update(
                task_epoch, description=f"Epoch {idx_epoch + 1} of {count_epoch}..."
            )
            task_train = progress.add_task("Training...", total=count_batch)
            for idx_batch, batch in enumerate(dataloader):
                model.train()
                output = model(**{k: v.to(device) for k, v in batch.items()})

                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()
                lr_scheduler.step()

                progress.update(
                    task_train,
                    description=f"{idx_batch + 1} of {count_batch}. Loss {output.loss}",
                )
                # if idx_batch % 4 == 0:
                #     print(output.logits.argmax(-1).cpu() - batch["labels"])
                progress.advance(task_train)
            progress.advance(task_epoch)

            if dataloader_eval:
                eval(model, dataloader=dataloader_eval, device=device)

        progress.refresh()
    return model.to(device_prev)


def pretrain_on_nsp(
    model_original: models.ModelMultipleChoice,
    *,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> models.ModelMultipleChoice:
    assert isinstance(model_original, models.ModelMultipleChoice)
    model = models.ModelNextSentencePrediction(model_original.bert)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PARAMS.PRETRAIN_NSP_LEARNING_RATE
    )

    count_epoch = 1
    num_training_steps = count_epoch * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    # accelerator = Accelerator()

    # dataloader, model, optimizer = typing_identity(accelerator.prepare)(dataloader, model, optimizer)
    device_prev = get_device_of_module(model)
    model = model.to(device)

    for param in model.bert.encoder.layer[:9].parameters():  # type: ignore
        param.requires_grad = False  # type: ignore

    count_batch = len(dataloader)

    with (PARAMS.progress(lambda: rich.progress.Progress()) as progress,):
        task_epoch = progress.add_task("Pretraining...", total=count_epoch)
        for idx_epoch in range(count_epoch):
            progress.update(
                task_epoch,
                description=f"Pretrain Epoch {idx_epoch + 1} of {count_epoch}...",
            )
            task_train = progress.add_task("Pretraining...", total=count_batch)
            for idx_batch, batch in enumerate(dataloader):
                model.train()
                output = model(**{k: v.to(device) for k, v in batch.items()})

                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()
                lr_scheduler.step()

                progress.update(
                    task_train,
                    description=f"{idx_batch + 1} of {count_batch}. Loss {output.loss}",
                )
                # if idx_batch % 4 == 0:
                #     print(output.logits.argmax(-1).cpu() - batch["labels"])
                progress.advance(task_train)
            progress.advance(task_epoch)
        progress.refresh()

    model = model.to(device_prev)
    return models.ModelMultipleChoice(model.bert)


def eval(
    model: Model,
    *,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Model:
    # accelerator = Accelerator()

    # dataloader, model, optimizer = typing_identity(accelerator.prepare)(dataloader, model, optimizer)

    with on_device(model, device) as model:

        model.eval()
        count_batch = len(dataloader)

        count_true = 0
        total = 0
        with (
            torch.no_grad(),
            PARAMS.progress(lambda: rich.progress.Progress()) as progress,
        ):
            task_eval = progress.add_task("Evaluation...", total=count_batch)
            for idx_batch, batch in enumerate(dataloader):

                output = model(
                    **{k: v.to(device) for k, v in batch.items() if k != "labels"}
                )
                count_true += (
                    len(batch["labels"])
                    - (
                        cast(torch.Tensor, output.logits).argmax(-1).cpu()
                        - batch["labels"]
                    ).count_nonzero()
                ).item()
                total += batch["labels"].shape[0]

                progress.update(
                    task_eval,
                    advance=1,
                    description=f"{idx_batch + 1} of {count_batch}. Accuracy {count_true} of {total} ({count_true / total * 100 if total else 0.0:.2f}%)",
                )

        # return accelerator.unwrap_model(model)
        progress.refresh()

    return model


__all__ = ["get_pretrained", "train"]
