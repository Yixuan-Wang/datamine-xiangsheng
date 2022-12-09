import contextlib
import dataclasses
from tkinter import CURRENT
from typing import Callable, Optional
import rich.progress


@dataclasses.dataclass
class __Params:
    MODEL_PRETRAINED: str = "bert-base-chinese"
    MODEL_MAX_TOKENS: int = 128
    BATCH_SIZE: int = 16
    CURRENT_PROGRESS_BAR: Optional[rich.progress.Progress] = None
    LEARNING_RATE: float = 5e-5
    PRETRAIN_NSP_LEARNING_RATE: float = 5e-5
    CONTEXT_LENGTH: int = 9
    FREEZE_LAYERS: int = 9
    DATA_SIZE: int = 1500
    VALID_SIZE: Optional[int] = 1000

    @contextlib.contextmanager
    def progress(self, init: Callable[[], rich.progress.Progress]):
        if self.CURRENT_PROGRESS_BAR is None:
            with init() as progress:
                try:
                    self.CURRENT_PROGRESS_BAR = progress
                    yield progress
                finally:
                    self.CURRENT_PROGRESS_BAR = None
        else:
            try:
                yield self.CURRENT_PROGRESS_BAR
            finally:
                pass


PARAMS = __Params()

__all__ = ["PARAMS"]
