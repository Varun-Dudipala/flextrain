"""Small shared helpers: seeding and rank-aware logging."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Union

import torch

_HANDLER_FLAG = "_flextrain_handler"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy (if installed) and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % 2**32)
    except ImportError:  # pragma: no cover
        pass


def setup_logging(level: str = "INFO", rank: int = 0, log_file: Optional[Union[str, Path]] = None) -> None:
    """Configure the ``flextrain`` logger unless the application already configured logging.

    Non-zero ranks only emit warnings and errors (unless ``level`` is DEBUG), so an 8-GPU
    job does not print every line eight times. Each line carries its rank.
    """
    pkg_logger = logging.getLogger("flextrain")
    fmt = logging.Formatter(f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    console_level = level if rank == 0 or level == "DEBUG" else "WARNING"

    app_configured = bool(logging.getLogger().handlers)
    has_ours = any(getattr(h, _HANDLER_FLAG, False) for h in pkg_logger.handlers)
    if not app_configured and not has_ours:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(console_level)
        setattr(handler, _HANDLER_FLAG, True)
        pkg_logger.addHandler(handler)
        pkg_logger.propagate = False
    pkg_logger.setLevel(level)

    if log_file is not None:
        log_file = Path(log_file)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in pkg_logger.handlers):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(fmt)
            setattr(file_handler, _HANDLER_FLAG, True)
            pkg_logger.addHandler(file_handler)


def teardown_file_logging() -> None:
    pkg_logger = logging.getLogger("flextrain")
    for handler in list(pkg_logger.handlers):
        if isinstance(handler, logging.FileHandler) and getattr(handler, _HANDLER_FLAG, False):
            pkg_logger.removeHandler(handler)
            handler.close()
