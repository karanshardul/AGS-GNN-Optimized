"""Simple timing utilities for precomputation, sampling, and training."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def timed_section(name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        line = f"[AGS timing] {name}: {elapsed:.4f}s"
        print(line, flush=True)
        logger.info("%s", line)


def log_timing(name: str, seconds: float) -> None:
    line = f"[AGS timing] {name}: {seconds:.4f}s"
    print(line, flush=True)
    logger.info("%s", line)
