"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from typing import Tuple
import pytorch_lightning as pl
from torch import Tensor
from torch.optim import Optimizer
from transformers import BertForMaskedLM, BertModel, BertTokenizer
from torch.nn import functional as F


class Frutifier(pl.LightningModule):
    # fruitify into these!
    FRUITS: Tuple[str] = ('apple', 'banana', 'strawberry', 'orange', 'grape')

    def __init__(self, k: int):
        super().__init__()
        # -- hyper params --- #
        self.k = k

    @classmethod
    def frutify(cls, desc: str, tokenizer: BertTokenizer) -> Tuple[Tuple[str, float]]:
        """
        Given a description, returns a list of fruits that best matches with the description.
        """
        raise NotImplementedError

    def forward(self, X: Tensor) -> Tensor:
        """
        forward defines the prediction/inference actions.
        """
        raise NotImplementedError

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Computes the loss with respect to the bach and returns the loss.
        """
        raise NotImplementedError

    def configure_optimizers(self) -> Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # TODO: what optimizer did the authors used?
        pass


class MonoLingFruit(Frutifier):
    """
    Eng-Eng monolingual fruitifier.
    """

    def __init__(self, bert_mlm: BertForMaskedLM, k: int):
        super().__init__(k)
        self.bert_mlm = bert_mlm  # this is the only layer we need, as far as MonoLing RD is concerned

    @classmethod
    def frutify(cls, desc: str, tokenizer: BertTokenizer) -> Tuple[Tuple[str, float]]:
        # TODO: Get use of cls.FRUITS
        pass

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (N, L) int tensor
        :return: (N, K, |S|); (num samples, k, the size of the vocabulary of subwords)
        """
        # TODO: Get use of bert_mlm.mbert(), bert_mlm.cls(). Help: examples/bert_mlm.py
        pass

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        :param batch: ((N, L), (N, |S|)). The second element is a one-hot vector.
        :param batch_idx: the index of the batch
        :return: (1,); the loss for this batch
        """
        X, Y = batch
        # TODO: use F.cross_entropy() to compute the loss, and return it. Help: examples/cross_entropy.py


class UnalignedCrossLingFruit(Frutifier):
    """
    Kor-Eng unaligned cross-lingual fruitifier.
    """
    def __init__(self, mbert: BertModel, k: int):
        super().__init__(k)
        # may want to add language embedding here...
        self.mbert = mbert  # we are using multi-lingual bert for this.
        # TODO: we need adding more layers here...

    @classmethod
    def frutify(cls, desc: str, tokenizer: BertTokenizer) -> Tuple[Tuple[str, float]]:
        # TODO: Get use of cls.FRUITS
        pass

    def forward(self, X: Tensor) -> Tensor:
        # TODO: ...?
        pass

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        X, Y = batch
        # TODO: use F.cross_entropy to compute the loss, and return it
