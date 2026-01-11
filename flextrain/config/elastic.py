"""Elastic scaling configuration."""

from dataclasses import dataclass
from typing import Optional
from .base import BaseConfig


@dataclass
class ElasticConfig(BaseConfig):
    """Configuration for elastic scaling."""

    min_nodes: int = 1
    max_nodes: int = 1
    nproc_per_node: int = 1

    rendezvous_backend: str = "c10d"
    rendezvous_endpoint: Optional[str] = None
    rendezvous_id: Optional[str] = None

    allow_elastic_scaling: bool = False
    scale_cooldown_seconds: int = 300

    max_restarts: int = 3
    restart_on_failure: bool = True

    heartbeat_interval_seconds: float = 30.0
    heartbeat_timeout_seconds: float = 300.0
    node_timeout_seconds: float = 900.0

    handle_preemption: bool = True
    preemption_grace_period_seconds: int = 60

    master_addr: str = "localhost"
    master_port: int = 29500

    def validate(self) -> None:
        if self.min_nodes < 1:
            raise ValueError("min_nodes must be at least 1")
        if self.max_nodes < self.min_nodes:
            raise ValueError("max_nodes must be >= min_nodes")
