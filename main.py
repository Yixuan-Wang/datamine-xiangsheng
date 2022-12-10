from datetime import datetime
from typing import Optional, cast

import torch
from rich import inspect, print
from transformers import logging as t_logging

import predict
import preprocess
import tune
from params import PARAMS
import models
import logging


import typer

app = typer.Typer()


@app.command(help="Train model on the `train` dataset.")
def train(
    pretrain: bool = typer.Option(False, help="Whether to include pretrain step."),
    save_model: Optional[str] = typer.Option(
        None,
        "--save",
        help="Whether to save the trained model and its name (dist/model.().(timestamp).pt).",
    ),
    context: int = typer.Option(
        9, help="How many utterances are included in the question as context."
    ),
    freeze_layers: int = typer.Option(
        9, help="How many BERT layers are freezed during (pre)training."
    ),
    data_size: int = typer.Option(1500, help="Size of training data."),
    learning_rate: float = typer.Option(
        5e-5, "--lr", help="Learning rate of training step."
    ),
    batch_size: int = typer.Option(16, help="Batch size of training step."),
):
    PARAMS.BATCH_SIZE = batch_size
    PARAMS.CONTEXT_LENGTH = context
    PARAMS.DATA_SIZE = data_size
    PARAMS.FREEZE_LAYERS = freeze_layers
    PARAMS.LEARNING_RATE = learning_rate

    timestamp = datetime.now().strftime("%m%dT%H%M%S")

    # Load pretrained BERT from Huggingface Hub
    tokenizer, bert = tune.get_pretrained()
    model = models.ModelMultipleChoice(cast(models.BertModel, bert))

    # A pretrain step on a next sentence prediction task
    if pretrain:
        dataloader_pretrain = preprocess.get_nsp_dataloader(tokenizer=tokenizer)
        model = tune.pretrain_on_nsp(
            model, dataloader=dataloader_pretrain, device=torch.device(0)
        )

    dataloader_train = preprocess.get_dataloader("train", tokenizer=tokenizer)
    dataloader_valid = preprocess.get_dataloader("valid", tokenizer=tokenizer)

    # Train (Finetune)
    model = tune.train(
        model=model,
        dataloader=dataloader_train,
        device=torch.device(0),
        dataloader_eval=dataloader_valid,
    )

    timestamp_to = datetime.now().strftime("%m%dT%H%M%S")

    if save_model is not None:
        torch.save(model, f"models/model.{save_model}.{timestamp}.{timestamp_to}.pt")


@app.command(help="Evaluate model with `valid` dataset.")
def eval(
    name: str,
    valid_size: Optional[int] = typer.Option(
        None,
        help="Sampling amount of validation on samples. Leave `None` for no sampling.",
    ),
):
    print(f"[blue bold]eval: [underline]{name}")
    PARAMS.VALID_SIZE = valid_size

    tokenizer, model = tune.get_pretrained(name)
    dataloader = preprocess.get_dataloader("valid", tokenizer=tokenizer)

    # Evaluate the model
    model = tune.eval(model, dataloader=dataloader, device=torch.device(0))


@app.command(help="Predict labels on the `test` dataset.")
def test(name: str):
    print(f"[green bold]test: [underline]{name}")
    tokenizer, model = tune.get_pretrained(name)
    dataloader = preprocess.get_test_dataloader(tokenizer=tokenizer)

    with open("data/test_label.json", "w") as buf:
        predict.predict(
            model, buffer=buf, dataloader=dataloader, device=torch.device(0)
        )


if __name__ == "__main__":
    # Suppress `huggingface` warnings
    t_logging.set_verbosity_error()
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    # CLI entrance
    app()
