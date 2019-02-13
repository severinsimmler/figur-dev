from . import corpus
from . import trainer
from .model import Model, URL

from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter



def train(directory: str, optimize: bool = True, features: list,
          metric: str = "micro-average f1-score", learning_rate: float = .1,
          mini_batch_size: int = 32, epochs: int = 10, hidden_size: int = 256,
          crf: bool = True, rnn: bool = True, rnn_layers: bool = True,
          dropout: float = .0, word_dropout: float = .05,
          locked_dropout: float = .5):
    """Train and optimize a model for recognizing named entities in literary texts.

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
    # Optional: Hyperparameter optimization:
    if optimize:
        optimize(".")
    # 3. Train model:
    trainer.train("figur-recognition/model",
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


def optimize(directory):
    data = corpus.load(directory)
    # Define search space:
    search_space = SearchSpace()
    # Collect embeddings:
    fasttext = trainer.utils.collect_features(["fasttext"])
    bert = trainer.utils.collect_features(["bert"])
    flair = trainer.utils.collect_features(["flair-forward", "flair-backward"])
    # Add to search space:
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
        fasttext,
        bert,
        flair
    ])
    # Add other parameters:
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])
    # Create parameter selector:
    param_selector = SequenceTaggerParamSelector(
        corpus=data,
        tag_type="ner",
        base_path="figur-recognition/optimization",
        max_epochs=10,
        training_runs=3)
    # Start the optimization:
    param_selector.optimize(search_space, max_evals=100)


def tag(text: str, model: Model = None):
    if not model:
        model = Model(URL)
    document = list()
    for sentence in document:
        sentence = Sentence(sentence)
         prediction = model.predict(sentence)
         document.append(prediction)
    return pd.DataFrame(document)
