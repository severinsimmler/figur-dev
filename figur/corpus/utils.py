"""
figur.corpus.utils
~~~~~~~~~~~~~~~~~~

This modules provides helper functions for this package.
"""

import random
from pathlib import Path


def train_dev_test(filepath: str, dev: float, test: float, seed: int):
    """Split corpus into train–dev–test datasets.
    """
    # Do not use more than 50% of the corpus for dev/test:
    assert dev + test < .5
    train = 1 - (test + dev)
    dev = train + dev
    # Parse data:
    data = list(parse_corpus(filepath))
    # Sort list:
    data.sort()
    # Set random seed:
    random.seed(seed)
    # Shuffle data:
    random.shuffle(data)
    # Set split indices:
    a = int(train * len(data))
    b = int(dev * len(data))
    # Split data:
    return {"train": data[:a],
            "dev":  data[a:b],
            "test": data[b:]}


def parse_corpus(filepath: str):
    """Parse corpus per instance.
    """
    with Path(filepath).open("r", encoding="utf-8") as file:
        for sentence in file.read().split("\n\n"):
            yield sentence
