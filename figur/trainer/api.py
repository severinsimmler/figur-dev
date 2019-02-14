"""
figur.trainer.api
~~~~~~~~~~~~~~~~~

The high-level API for this package.
"""

from pathlib import Path

import flair
import torch

from figur.corpus.model import Corpus
from . import model
from . import utils


def train(directory: str, features: list, metric: str = "micro-average f1-score",
          learning_rate: float = .1, mini_batch_size: int = 32,
          epochs: int = 10, **kwargs):
    """Train a model for named entity recognition.
    """
    # Construct corpus object:
    data = Corpus(directory).fetch("train.tsv",
                                   "dev.tsv",
                                   "test.tsv")
    # Collect features:
    features = list(utils.collect_features(features))
    # Construct trainer object:
    trainer = model.Trainer(data, features, **kwargs)
    # Train model:
    trainer.train(directory,
                  metric=metric,
                  learning_rate=learning_rate,
                  mini_batch_size=mini_batch_size,
                  epochs=epochs)
