"""The checkpoint directory's manifest: the commit log of the lineage being written there.

A directory and a step number do not identify a checkpoint. Reruns, rollbacks and restarts all
write ``checkpoint_step<N>.pt`` into the same directory, so "the highest step" can belong to an
abandoned run, and retention by step would delete a new run's first checkpoints in favour of an
old run's later ones. The manifest (``manifest.json``, replaced atomically) records what the
directory listing cannot:

* **lineage** - an id for one continuous training history. A fresh start, or a start from an
  explicit ``resume_path``, begins a new lineage; resuming continues one. Only checkpoints
  listed here belong to the current lineage; retention and auto-resume look at nothing else,
  and never delete a file this lineage did not write.
* **commit point** - a checkpoint counts once its entry is in the manifest, which is written
  after the checkpoint file and its digest are durable. A crash before that leaves a file that
  is ignored (and overwritten when that step is saved again).
* **origin / launch** - the checkpoint a lineage was branched from and the launch that is
  writing it, which is how a restart knows to continue the lineage instead of re-applying
  ``resume_path`` and throwing away everything trained since.
* **writer / generation** - a fencing token. A process that finds another writer's token where
  it expected its own stops with ``CheckpointConflictError`` instead of interleaving its
  checkpoints with another job's.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = 1


class ManifestError(ValueError):
    """The manifest exists but cannot be understood."""


class CheckpointConflictError(RuntimeError):
    """Another process is writing checkpoints to the same directory."""


class NoUsableCheckpointError(RuntimeError):
    """The lineage has checkpoints, but every one of them is corrupted."""


@dataclass
class ManifestEntry:
    step: int
    file: str  # name relative to the checkpoint directory
    nbytes: int = 0
    time: float = 0.0


@dataclass
class Manifest:
    lineage: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: float = field(default_factory=time.time)
    origin: Optional[str] = None  # checkpoint this lineage was branched from, if any
    launch_id: Optional[str] = None  # the launch currently writing this lineage
    writer: str = ""
    generation: int = 0
    checkpoints: List[ManifestEntry] = field(default_factory=list)

    def sorted_entries(self) -> List[ManifestEntry]:
        return sorted(self.checkpoints, key=lambda e: e.step)

    def to_bytes(self) -> bytes:
        data = asdict(self)
        data["format"] = MANIFEST_FORMAT
        data["checkpoints"] = [asdict(e) for e in self.sorted_entries()]
        return (json.dumps(data, indent=2) + "\n").encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> Manifest:
        try:
            data = json.loads(raw)
            if data.get("format", 0) > MANIFEST_FORMAT:
                raise ManifestError(f"manifest format {data['format']} is newer than supported ({MANIFEST_FORMAT})")
            entries = [ManifestEntry(step=int(e["step"]), file=str(e["file"]), nbytes=int(e.get("nbytes", 0)),
                                     time=float(e.get("time", 0.0))) for e in data.get("checkpoints", [])]
            return cls(lineage=str(data["lineage"]), created=float(data.get("created", 0.0)),
                       origin=data.get("origin"), launch_id=data.get("launch_id"),
                       writer=str(data.get("writer", "")), generation=int(data.get("generation", 0)),
                       checkpoints=entries)
        except ManifestError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise ManifestError(f"unreadable manifest: {type(e).__name__}: {e}") from e
