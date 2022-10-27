# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import os
import time
import warnings

import torch
from pytorch_lightning import LightningModule, loggers as pl_loggers, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.classification.accuracy import accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

warnings.filterwarnings("ignore")  # Disable data logger warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)  # Disable GPU/TPU prints


def parse_args():
    parser = argparse.ArgumentParser(description="train mnist")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="dir to place tensorboard logs from all trials",
    )
    parser.add_argument(
        "--hidden_size_1", type=int, required=True, help="hidden size layer 1"
    )
    parser.add_argument(
        "--hidden_size_2", type=int, required=True, help="hidden size layer 2"
    )
    parser.add_argument(
        "--learning_rate", type=float, required=True, help="learning rate"
    )
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
    parser.add_argument(
        "--dropout", type=float, required=True, help="dropout probability"
    )
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    return parser.parse_args()


args = parse_args()

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())


class MnistModel(LightningModule):
    def __init__(self):
        super().__init__()

        # Tunable parameters
        self.hidden_size_1 = args.hidden_size_1
        self.hidden_size_2 = args.hidden_size_2
        self.learning_rate = args.learning_rate
        self.dropout = args.dropout
        self.batch_size = args.batch_size

        # Set class attributes
        self.data_dir = PATH_DATASETS

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Create a PyTorch model
        layers = [nn.Flatten()]
        width = channels * width * height
        hidden_layers = [self.hidden_size_1, self.hidden_size_2]
        num_params = 0
        for hidden_size in hidden_layers:
            if hidden_size > 0:
                layers.append(nn.Linear(width, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
                num_params += width * hidden_size
                width = hidden_size
        layers.append(nn.Linear(width, self.num_classes))
        num_params += width * self.num_classes

        # Save the model and parameter counts
        self.num_params = num_params
        self.model = nn.Sequential(*layers)  # No need to use Relu for the last layer

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_acc", acc, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_val = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)


def run_training_job():

    mnist_model = MnistModel()

    # Initialize a trainer
    logger = pl_loggers.TensorBoardLogger(args.log_path)
    trainer = Trainer(
        logger=logger,
        log_every_n_steps=1,
        gpus=AVAIL_GPUS,
        max_epochs=args.epochs,
        enable_progress_bar=False,
        deterministic=True,  # Do we want a bit of noise?
        default_root_dir=args.log_path,
    )
    logger.save()

    print(f"Logging to path: {args.log_path}.")

    # Train the model and log time
    start = time.time()
    trainer.fit(model=mnist_model)
    end = time.time()
    train_time = end - start

    # Compute the validation accuracy
    val_accuracy = trainer.validate()[0]["val_acc"]

    # Log the number of model parameters
    num_params = trainer.model.num_params

    # Print outputs
    print(
        f"train time: {train_time}, val acc: {val_accuracy}, num_params: {num_params}"
    )


if __name__ == "__main__":
    run_training_job()
