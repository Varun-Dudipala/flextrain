"""Train a GPT-2 style language model with FlexTrain.

The default config trains a tiny byte-level GPT-2 on this repository's own source code,
so it runs anywhere (CPU included) with no downloads, and the loss visibly drops.

Single process:
    python examples/gpt2_training/train_gpt2.py

Two workers with elastic restarts (kill one mid-run and watch it resume):
    flextrain launch -c examples/gpt2_training/config.yaml --nproc-per-node 2 \
        examples/gpt2_training/train_gpt2.py

GPT-2 small on a GPU:
    python examples/gpt2_training/train_gpt2.py --set model.size=small --set data.seq_len=512
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import create_gpt2_model  # noqa: E402

from flextrain import Trainer, load_config  # noqa: E402
from flextrain.config import CONFIG_ENV_VAR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


class ByteTextDataset(Dataset):
    """Byte-level LM dataset: non-overlapping ``seq_len`` windows over the concatenated text."""

    def __init__(self, files: List[Path], seq_len: int):
        if not files:
            raise ValueError("no training text files found")
        data = b"\n".join(Path(f).read_bytes() for f in files)
        self.tokens = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        self.seq_len = seq_len
        self.num_windows = len(self.tokens) // seq_len

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int):
        chunk = self.tokens[index * self.seq_len : (index + 1) * self.seq_len]
        return {"input_ids": chunk, "labels": chunk}  # the model shifts labels internally


def resolve_files(patterns: List[str]) -> List[Path]:
    if not patterns:
        patterns = [str(REPO_ROOT / "flextrain" / "**" / "*.py")]
    files = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern, recursive=True)})
    return [f for f in files if f.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help=f"YAML config (default: ${CONFIG_ENV_VAR} or {DEFAULT_CONFIG.name})")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    args = parser.parse_args()

    config_path = args.config or (None if os.environ.get(CONFIG_ENV_VAR) else DEFAULT_CONFIG)
    config = load_config(config_path, overrides=args.set)
    logging.getLogger("flextrain").setLevel(config.main.log_level)

    model_cfg = dict(config.model)
    size = model_cfg.pop("size", "tiny")
    seq_len = int(config.data.get("seq_len", 128))
    model_cfg.setdefault("max_position_embeddings", max(seq_len, 256))
    model = create_gpt2_model(size, **model_cfg)

    dataset = ByteTextDataset(resolve_files(config.data.get("paths", [])), seq_len)
    n_eval = max(1, len(dataset) // 10)
    train_set = Subset(dataset, range(len(dataset) - n_eval))
    eval_set = Subset(dataset, range(len(dataset) - n_eval, len(dataset)))

    with Trainer(model, config, train_set, eval_dataset=eval_set) as trainer:
        if trainer.is_main_process:
            print(f"GPT-2 {size}: {model.count_parameters() / 1e6:.2f}M params, "
                  f"{len(train_set)} train / {len(eval_set)} eval windows of {seq_len} bytes")
        result = trainer.train()
        if trainer.is_main_process:
            print(f"{result.status} at step {result.global_step}; last checkpoint: {result.last_checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
