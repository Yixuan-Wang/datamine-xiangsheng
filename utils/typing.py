from __future__ import annotations
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Any, Union, TypeVar, cast
from typing_extensions import ParamSpec, TypeVarTuple, Unpack
from collections.abc import Callable
import torch.nn

T = TypeVar("T")
P = TypeVar("P")
Ts = TypeVarTuple("Ts")

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
Model = torch.nn.Module

def typing_identity(call: Callable) -> Callable[[Unpack[Ts]], tuple[Unpack[Ts]]]:
    return call

__all__ = [
    "Model",
    "Tokenizer",
    "typing_identity",
]