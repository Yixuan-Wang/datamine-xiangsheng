import torch
import torch.nn

from utils.typing import *
import contextlib


def get_device_of_module(model: Model) -> torch.device:
    return next(model.parameters()).device


@contextlib.contextmanager
def on_device(model: Model, device: torch.device):
    original_device = get_device_of_module(model)
    try:
        yield model.to(device)
    finally:
        model.to(original_device)


__all__ = ["get_device_of_module"]
