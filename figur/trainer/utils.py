"""
figur.trainer.utils
~~~~~~~~~~~~~~~~~~~

This module provides helper functions for this package.
"""

from flair.embeddings import WordEmbeddings, BertEmbeddings, FlairEmbeddings

from . import model


def collect_features(embeddings):
    for embedding in embeddings:
        if embedding in {"fasttext"}:
            yield WordEmbeddings("de")
        elif embedding in {"bert"}:
            yield BertEmbeddings("bert-base-multilingual-cased")
        elif embedding in {"flair-forward"}:
            yield FlairEmbeddings("german-forward")
        elif embedding in {"flair-backward"}:
            yield FlairEmbeddings("german-backward")
