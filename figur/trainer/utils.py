"""
figur.trainer.utils
~~~~~~~~~~~~~~~~~~~

This module provides helper functions for this module.
"""
import flair
import torch
from flair.embeddings import WordEmbeddings, BertEmbeddings, FlairEmbeddings


def collect_features(embeddings, gpu):
    flair.device = torch.device("cpu")
    print(flair.device)
    mapping = {"fasttext": WordEmbeddings("de"),
               "bert": BertEmbeddings("bert-base-multilingual-cased"),
               "flair-forward": FlairEmbeddings("german-forward"),
               "flair-backward": FlairEmbeddings("german-backward")}
    for embedding in embeddings:
        yield mapping[embedding]
