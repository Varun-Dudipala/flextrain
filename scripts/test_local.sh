#!/bin/bash
# Quick test to verify FlexTrain installation

echo "Testing FlexTrain installation..."

cd "$(dirname "$0")/.."
pip install -e . -q

python3 -c "
from flextrain import Config, __version__
print(f'FlexTrain {__version__} installed successfully!')

config = Config()
print(f'Default config created: {config.main.experiment_name}')
"

echo "Done!"
