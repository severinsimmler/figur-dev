"""
figur.corpus.api
~~~~~~~~~~~~~~~~

The high-level API for this module.
"""

from pathlib import Path

from . import model
from . import utils


def process(directory: str, suffix: str = ".xml"):
    """Process a directory of XML files.
    """
    for filepath in Path(directory).rglob("*.xml"):
        yield model.Document(str(filepath))


def export(directory: str, suffix: str = ".xml", filepath: str = "corpus.tsv"):
    """Export a directory of XML files to training data.
    """
    with Path(filepath).open("w", encoding="utf-8") as file:
        for document in process(directory, suffix):
            for sentence in document.sentences:
                for n, (token, label) in enumerate(sentence):
                    file.write(f"{n}\t{token}\t{label}\n")
                file.write("\n")


def split(filepath: str, dev: float = .1, test: float = .1, seed: int = 23):
    """Split corpus file into train–test–dev data sets.
    """
    data = utils.train_dev_test(filepath, dev, test, seed)
    for name, instances in data.items():
        with Path(f"{name}.tsv").open("w", encoding="utf-8") as file:
            file.write("\n\n".join(instances))


def load(train: str, dev: str, test: str):
    """Load corpus data sets.
    """
    directory = Path(train).parent
    corpus = model.Corpus(directory)
    return corpus.fetch(train, dev, test)
