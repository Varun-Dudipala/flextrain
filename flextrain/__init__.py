"""FlexTrain - fault-tolerant distributed training for PyTorch."""

__version__ = "0.2.0"

from flextrain.config import Config, ConfigError, load_config  # noqa: E402
from flextrain.core.trainer import DistributedTrainer, Trainer, TrainResult  # noqa: E402
from flextrain.utils import set_seed  # noqa: E402

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "Trainer",
    "DistributedTrainer",
    "TrainResult",
    "set_seed",
    "__version__",
]
