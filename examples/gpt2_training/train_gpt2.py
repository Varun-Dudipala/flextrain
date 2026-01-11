"""GPT-2 training example with FlexTrain."""

import torch
from torch.utils.data import Dataset

from flextrain import Config, DistributedTrainer
from .model import create_gpt2_model


class RandomTextDataset(Dataset):
    """Random dataset for testing."""

    def __init__(self, vocab_size=50257, seq_length=512, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def main():
    # Create config
    config = Config()
    config.training.batch_size = 8
    config.training.learning_rate = 3e-4
    config.training.max_steps = 1000
    config.training.log_interval = 10
    config.training.save_interval = 100

    # Create model and dataset
    model = create_gpt2_model("small")
    dataset = RandomTextDataset()

    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        config=config,
        train_dataset=dataset,
    )

    # Train
    results = trainer.train()
    print(f"Training complete: {results}")


if __name__ == "__main__":
    main()
