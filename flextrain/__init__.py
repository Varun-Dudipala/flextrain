"""FlexTrain - Fault-tolerant distributed training framework."""

__version__ = "0.1.0"

from flextrain.config import Config, load_config
from flextrain.core.trainer import DistributedTrainer

__all__ = ["Config", "load_config", "DistributedTrainer", "__version__"]
