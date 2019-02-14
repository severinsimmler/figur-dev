"""
figur.trainer.utils
~~~~~~~~~~~~~~~~~~~

This module provides helper functions for this module.
"""
import flair
import torch
from flair.embeddings import WordEmbeddings, BertEmbeddings, FlairEmbeddings
from . import model


def collect_features(embeddings, gpu):
    model._set_device(gpu)
    for embedding in embeddings:
        if embedding in {"fasttext"}:
            yield WordEmbeddings("de")
        elif embedding in {"bert"}:
            yield BertEmbeddings("bert-base-multilingual-cased")
