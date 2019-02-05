from pathlib import Path

from . import model


def process(directory: str, suffix: str = ".xml"):
    for filepath in Path(directory).rglob("*.xml"):
        yield model.Document(str(filepath))


def export(directory: str, suffix: str = ".xml"):
    with open("corpus.tsv", "w", encoding="utf-8") as file:
        for document in process(directory, suffix):
            for sentence in document.sentences:
                for n, (token, label) in enumerate(sentence):
                    row = f"{n}\t{token}\t{label}\n"
                    file.write(row)
                file.write("\n")
