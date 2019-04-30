from dataclasses import dataclass


@dataclass
class Model:
    filepath: str

    def __post_init__(self):
        if "https" in self.filepath:
            # Download model:
            self.filepath = utils.download_model(self.filepath)
        self.tagger = SequenceTagger.load_from_file(self.filepath)

    def predict(self, sentence: str):    
        self.tagger.predict(Sentence(sentence))
        return sentence.to_tagged_string()

    def summary(self, input_size):
        #https://github.com/sksq96/pytorch-summary
        pass
