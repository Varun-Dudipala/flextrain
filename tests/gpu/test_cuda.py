"""CUDA-only paths: pinned-memory snapshots, mixed precision, GPU resume.

Skipped automatically without a GPU (CI is CPU-only). Run on a GPU box / Colab with:
    pytest tests/gpu -q
"""

import pytest
import torch

from flextrain import Trainer
from flextrain.checkpoint import SnapshotBuffers
from tests.conftest import RegressionDataset, TinyModel

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def test_snapshot_from_cuda_is_pinned_and_isolated():
    buffers = SnapshotBuffers(pin_memory=True)
    live = {"w": torch.randn(4096, device="cuda"), "nested": {"b": torch.ones(8, device="cuda")}}
    snap = buffers.snapshot(live)
    assert snap["w"].device.type == "cpu" and snap["w"].is_pinned()
    torch.testing.assert_close(snap["w"], live["w"].cpu())
    live["w"].add_(1.0)
    assert not torch.equal(snap["w"], live["w"].cpu())
    again = buffers.snapshot(live)
    assert again["w"].data_ptr() == snap["w"].data_ptr()  # pinned buffers reused


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_mixed_precision_training(make_config, dtype):
    if dtype == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported on this GPU")
    torch.manual_seed(0)
    cfg = make_config(40, distributed={"mixed_precision": True, "mixed_precision_dtype": dtype},
                      training={"log_interval": 10, "learning_rate": 3e-2})
    trainer = Trainer(TinyModel(dropout=0.0), cfg, RegressionDataset())
    assert trainer.device.type == "cuda"
    assert trainer.amp_dtype == (torch.bfloat16 if dtype == "bf16" else torch.float16)
    assert (trainer.scaler is not None) == (dtype == "fp16")  # loss scaling only for fp16
    result = trainer.train()
    losses = [e["loss"] for e in trainer.metrics_logger.read() if "loss" in e]
    assert result.status == "completed" and losses[-1] < losses[0]


def test_async_checkpoint_resume_on_gpu(make_config):
    torch.manual_seed(0)
    first = Trainer(TinyModel(), make_config(10), RegressionDataset())
    first.train()
    first.close()
    resumed = Trainer(TinyModel(), make_config(20), RegressionDataset())
    result = resumed.train()
    assert resumed.resumed_from.endswith("checkpoint_step00000010.pt")
    assert result.global_step == 20 and all(p.device.type == "cuda" for p in resumed.model.parameters())
    resumed.close()
