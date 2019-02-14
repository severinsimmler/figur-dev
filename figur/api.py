from pathlib import Path

from figur import corpus
from figur import trainer
from figur.model.api import Model
from figur.model.utils import URL

from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter



def train(directory: str, features: list, optimal: bool = True,
          metric: str = "micro-average f1-score", learning_rate: float = .1,
          mini_batch_size: int = 32, epochs: int = 10, hidden_size: int = 256,
          crf: bool = True, rnn: bool = True, rnn_layers: int = 1,
          dropout: float = .0, word_dropout: float = .05,
          locked_dropout: float = .5):
    """Train and optimize a model for recognizing named entities in literary texts.

    Parameters:
        directory:
        optimal:
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
    root = Path("figur-recognition")
    if not root.exists():
        root.mkdir()
    # 1. Export to a single corpus file:
    corpus.export(directory, filepath=Path("figur-recognition", "corpus.tsv"))

    # 2. Split into train–dev–test data sets:
    corpus.split(Path("figur-recognition", "corpus.tsv"))

    # Optional: Hyperparameter optimization:
    if optimal:
        optimize(".")
        # TODO: parse output file and set optimal parameters

    # 3. Train model:
    trainer.train(Path("figur-recognition"),
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

    # 4. Load and return the best model:
    path = Path("figur-model", "best-model.pt")
    return Model(path)


def optimize(directory):
    """Hyperparameter optimization.
    """
    # 1. Load corpus:
    data = corpus.load(directory)

    # 2. Define search space:
    space = SearchSpace()

    # 3. Collect embeddings:
    fasttext = trainer.utils.collect_features(["fasttext"])
    bert = trainer.utils.collect_features(["bert"])
    flair = trainer.utils.collect_features(["flair-forward", "flair-backward"])

    # 4. Add to search space:
    space.add(Parameter.EMBEDDINGS,
              hp.choice,
              options=[fasttext, bert, flair])

    # 5. Add other parameter search spaces:
    space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
    space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
    space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
    space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
    space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])

    # 6. Create parameter selector:
    selector = SequenceTaggerParamSelector(corpus=data,
                                           tag_type="ner",
                                           base_path=Path("figur-recognition", "optimization"),
                                           max_epochs=10,
                                           training_runs=3)

    # 7. Start the optimization:
    selector.optimize(space, max_evals=100)


def tag(text: str, model: Model = None):
    """Tag named entities in a text.
    """
    # 1. If no model passe, load it:
    if not model:
        model = Model(URL)

    # 2. Segment text into sentences:
    sentences = utils.segment_sentences(text)

    # 3. Tag each sentence:
    for sentence in sentences:
        yield model.predict(sentence)
