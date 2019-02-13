from dataclasses import dataclass
import urllib.request

URL = ""

@dataclass
class Model:
    filepath: str

    def __post_init__(self):
        if "https" in self.filepath:
            # Download model:
            self.filepath = utils.download_model(self.filepath)
        self.tagger = SequenceTagger.load_from_file(self.filepath)

    def predict(self, sentence: Sentence):
        self.tagger.predict(sentence)
        return sentence.to_tagged_string()
