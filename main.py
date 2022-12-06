# %%
from datetime import datetime

from rich import inspect, print

# %%
timestamp = datetime.now().strftime("%m%dT%H%M%S")

# %%
import torch

# %%
import preprocess
import train
from params import PARAMS

# %%
tokenizer, model = train.get_pretrained()

# %%
dataloader_train = preprocess.get_dataloader("train", tokenizer=tokenizer)

# %%
model = train.train(model=model, dataloader=dataloader_train, device=torch.device(0))

# %%
dataloader_valid = preprocess.get_dataloader("valid", tokenizer=tokenizer)
model = train.eval(model=model, dataloader=dataloader_valid, device=torch.device(0))


timestamp_to = datetime.now().strftime("%m%dT%H%M%S")
torch.save(model, f"model.{timestamp}.{timestamp_to}.pt")
