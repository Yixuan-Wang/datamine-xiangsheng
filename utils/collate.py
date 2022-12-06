import torch.nn.functional
import torch
import operator
from datasets import Dataset  # type: ignore
from torch.utils.data import DataLoader

from params import PARAMS

def round_to_eight(x):
    return x if x % 8 == 0 else 8 * (x // 8 + 1)


def pad_to_with(ts, to, *, end, start):
    l = [
        torch.nn.functional.pad(t, (0, to - t.shape[-1]), value=end)
        if t.shape[-1] <= to
        else torch.nn.functional.pad(t[-to+1:], (1, 0), value=start)
        for t in ts
    ]
    return torch.stack(l)


def data_collator(features):
    pad_to = min(round_to_eight(
        max(map(lambda x: max(i.shape[-1] for i in x["input_ids"]), features))
    ), PARAMS.MODEL_MAX_TOKENS)


    for x in features:
        x["input_ids"] = pad_to_with(x["input_ids"], pad_to, start=101, end=0)
        x["attention_mask"] = pad_to_with(x["attention_mask"], pad_to, start=1, end=0)
        x["token_type_ids"] = pad_to_with(x["token_type_ids"], pad_to, start=0, end=1)

    return {
        k: torch.stack(list(map(operator.itemgetter(k), features)))
        for k in {"input_ids", "token_type_ids", "attention_mask", "labels"}
    }

__all__ = ["data_collator"]
