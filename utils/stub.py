import itertools
from collections.abc import Iterable, Callable, Generator
from typing import TypeVar, Optional

T = TypeVar("T")


def window(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    iters = itertools.tee(iterable, n)
    for idx, it in enumerate(iters):
        for _ in range(idx):
            next(it, None)
    return zip(*iters)


def track(
    *,
    hook_pre: Optional[Callable[[int, T], None]] = None,
    hook_post: Optional[Callable[[int, T], None]] = None
) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    def inner(it: Iterable[T]):
        for idx, each in enumerate(it):
            if hook_pre:
                hook_pre(idx, each)
            yield each
            if hook_post:
                hook_post(idx, each)

    return inner


__all__ = ["window"]
