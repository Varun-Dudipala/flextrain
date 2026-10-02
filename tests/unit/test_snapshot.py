"""Host snapshot tests."""

from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn

from flextrain.checkpoint import SnapshotBuffers, snapshot_to_cpu


def test_snapshot_is_isolated_from_later_mutation():
    live = {"w": torch.ones(4), "nested": {"b": torch.zeros(2)}, "list": [torch.ones(1)], "n": 3}
    snap = snapshot_to_cpu(live)
    live["w"].add_(1)
    live["nested"]["b"].add_(1)
    live["list"][0].add_(1)
    assert torch.equal(snap["w"], torch.ones(4))
    assert torch.equal(snap["nested"]["b"], torch.zeros(2))
    assert torch.equal(snap["list"][0], torch.ones(1))
    assert snap["n"] == 3


def test_preserves_container_types():
    Point = namedtuple("Point", "x y")
    live = OrderedDict(a=torch.ones(1), t=(torch.ones(1), 2), p=Point(torch.ones(1), 5))
    snap = snapshot_to_cpu(live)
    assert isinstance(snap, OrderedDict) and list(snap) == ["a", "t", "p"]
    assert isinstance(snap["t"], tuple) and snap["t"][1] == 2
    assert isinstance(snap["p"], Point) and snap["p"].y == 5


def test_tied_weights_stay_shared():
    emb = nn.Embedding(10, 4)
    head = nn.Linear(4, 10, bias=False)
    head.weight = emb.weight
    state = {"emb.weight": emb.weight.detach(), "head.weight": head.weight.detach()}
    snap = snapshot_to_cpu(state)
    assert snap["emb.weight"] is snap["head.weight"]


def test_buffers_are_reused_across_snapshots():
    buffers = SnapshotBuffers(reuse=True)
    first = buffers.snapshot({"w": torch.ones(8)})
    ptr = first["w"].data_ptr()
    second = buffers.snapshot({"w": torch.full((8,), 2.0)})
    assert second["w"].data_ptr() == ptr
    assert torch.equal(second["w"], torch.full((8,), 2.0))
    assert buffers.nbytes == 8 * 4


def test_buffer_reallocated_on_shape_change():
    buffers = SnapshotBuffers(reuse=True)
    buffers.snapshot({"w": torch.ones(8)})
    snap = buffers.snapshot({"w": torch.ones(3, 3)})
    assert snap["w"].shape == (3, 3)


def test_without_reuse_each_snapshot_is_fresh():
    buffers = SnapshotBuffers(reuse=False)
    a = buffers.snapshot({"w": torch.ones(8)})
    b = buffers.snapshot({"w": torch.ones(8)})
    assert a["w"].data_ptr() != b["w"].data_ptr()
    assert buffers.nbytes == 0


def test_non_contiguous_and_scalar_tensors():
    t = torch.arange(12.0).view(3, 4).t()
    snap = snapshot_to_cpu({"t": t, "step": torch.tensor(5.0)})
    assert torch.equal(snap["t"], t) and snap["step"].item() == 5.0
