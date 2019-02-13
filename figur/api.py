from . import corpus
from . import trainer
from .model import Model, URL


def train(directory: str, optimize: bool = True, features: list,
          metric: str = "micro-average f1-score", learning_rate: float = .1,
          mini_batch_size: int = 32, epochs: int = 10, hidden_size: int = 256,
          crf: bool = True, rnn: bool = True, rnn_layers: bool = True,
          dropout: float = .0, word_dropout: float = .05,
          locked_dropout: float = .5):
    """Train a model for recognizing named entities in literary texts.

    Parameters:
        directory:
        features:
        metric:
        learning_rate:
        mini_batch_size:
        epochs:
        hidden_size:
        crf:
        rnn:
        rnn_layers:
        dropout:
        word_dropout:
        locked_dropout:

    Returns:
        A model with which you can label a sequence of tokens.
    """
    # 1. Export to a single corpus file:
    corpus.export(directory, filepath="corpus.tsv")
    # 2. Split into train–dev–test data sets:
    corpus.split("corpus.tsv")
    # 3. Train model:
    trainer.train("figur-model",
                  features=features,
                  metric=metric,
                  learning_rate=learning_rate,
                  mini_batch_size=mini_batch_size,
                  epochs=epochs,
                  hidden_size=hidden_size,
                  crf=crf,
                  rnn=rnn,
                  rnn_layers=rnn_layers,
                  dropout=dropout,
                  word_dropout=word_dropout,
                  locked_dropout=locked_dropout)
    # 4. TODO: Summarize performance:
    #
    # 5. Load the best model:
    path = Path("figur-model", "best-model.pt")
    model = Model(path)
    # 6. Return the model:
    return model

def optimize(model: Model):



def label(text: str, model: Model = None):
    if not model:
        model = Model(URL)
    document = list()
    for sentence in document:
        sentence = Sentence(sentence)
         prediction = model.predict(sentence)
         document.append(prediction)
    return pd.DataFrame(document)
