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
    for embedding in embeddings:
        if embedding in {"fasttext"}:
            yield WordEmbeddings("de")
        elif embedding in {"bert"}:
            yield BertEmbeddings("bert-base-multilingual-cased")
