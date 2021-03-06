"""
figur.trainer.model
~~~~~~~~~~~~~~~~~~~

This module provides classes for the model trainer.
"""

from dataclasses import dataclass
from pathlib import Path

from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import TaggedCorpus
from flair.trainers import ModelTrainer
from flair.embeddings import StackedEmbeddings
from flair.models import SequenceTagger
from flair.training_utils import EvaluationMetric


@dataclass
class Trainer:
    corpus: TaggedCorpus
    features: list
    hidden_size: int = 256
    crf: bool = True
    rnn: bool = True
    rnn_layers: int = 1
    dropout: float = .0
    word_dropout: float = .05
    locked_dropout: float = .5

    @property
    def tags(self):
        return self.corpus.make_tag_dictionary(tag_type="ner")

    @property
    def embeddings(self):
        return StackedEmbeddings(embeddings=self.features)

    @property
    def tagger(self):
        return SequenceTagger(hidden_size=self.hidden_size,
                              embeddings=self.embeddings,
                              tag_dictionary=self.tags,
                              tag_type="ner",
                              use_crf=self.crf,
                              use_rnn=self.rnn,
                              rnn_layers=self.rnn_layers,
                              dropout=self.dropout,
                              word_dropout=self.word_dropout,
                              locked_dropout=self.locked_dropout)

    @property
    def trainer(self):
        return ModelTrainer(self.tagger,
                            self.corpus)

    def train(self, directory: str, metric: str = "micro-average f1-score",
              learning_rate: float = .1, mini_batch_size: int = 32,
              epochs: int = 10):
        metrics = {"micro-average accuracy": EvaluationMetric.MICRO_ACCURACY,
                   "micro-average f1-score": EvaluationMetric.MICRO_F1_SCORE,
                   "macro-average accuracy": EvaluationMetric.MACRO_ACCURACY,
                   "macro-average f1-score": EvaluationMetric.MACRO_F1_SCORE}
        assert metric in metrics
        self.trainer.train(Path(directory),
                           evaluation_metric=metrics[metric],
                           learning_rate=learning_rate,
                           mini_batch_size=mini_batch_size,
                           max_epochs=epochs)
