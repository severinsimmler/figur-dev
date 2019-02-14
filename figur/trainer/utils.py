"""
figur.trainer.utils
~~~~~~~~~~~~~~~~~~~

This module provides helper functions for this module.
"""
import flair
import torch
from flair.embeddings import WordEmbeddings, BertEmbeddings, FlairEmbeddings


def collect_features(embeddings, gpu):
    if gpu:
        flair.device = torch.device("gpu")
    else:
        flair.device = torch.device("cpu")
    values = {"fasttext": "de",
              "bert": "bert-base-multilingual-cased"}
    mapping = {"fasttext": WordEmbeddings,
               "bert": BertEmbeddings}
    for embedding in embeddings:
        yield mapping[embedding](values[embedding])
