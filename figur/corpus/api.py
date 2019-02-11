from pathlib import Path

from . import model


def process(directory: str, suffix: str = ".xml"):
    """Process a directory of XML files.
    """
    for filepath in Path(directory).rglob("*.xml"):
        yield model.Document(str(filepath))


def export(directory: str, suffix: str = ".xml", output: str = "corpus.tsv"):
    """Export a directory of XML files to training data.
    """
    with Path(output).open("w", encoding="utf-8") as corpusfile:
        for document in process(directory, suffix):
            for sentence in document.sentences:
                for n, (token, label) in enumerate(sentence):
                    file.write(f"{n}\t{token}\t{label}\n")
                file.write("\n")
