import dataclasses


@dataclasses.dataclass
class __Params():
    MODEL_PRETRAINED: str = "bert-base-chinese"
    MODEL_MAX_TOKENS: int = 128
    BATCH_SIZE: int = 16

PARAMS = __Params()

__all__ = ["PARAMS"]
