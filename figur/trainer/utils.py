from flair.embeddings import WordEmbeddings, BertEmbeddings, FlairEmbeddings


def collect_features(embeddings):
    mapping = {"fasttext": WordEmbeddings("de"),
               "bert": BertEmbeddings("bert-base-multilingual-cased"),
               "flair-forward": FlairEmbeddings("german-forward"),
               "flair-backward": FlairEmbeddings("german-backward")}
    for embedding in embeddings:
        yield mapping[embedding]
