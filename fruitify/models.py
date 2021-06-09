"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from typing import Tuple, List
import pytorch_lightning as pl
from torch import Tensor
from torch.optim import Optimizer
from transformers import BertForMaskedLM, BertModel, BertTokenizer
from torch.nn import functional as F


class Frutifier(pl.LightningModule):
    # fruitify into these!
    FRUITS: Tuple[str] = ('apple', 'banana', 'strawberry', 'orange', 'banana')

    def __init__(self, k: int):
        super().__init__()
        # -- hyper params --- #
        self.k = k

    @staticmethod
    def frutify(desc: str, tokenizer: BertTokenizer) -> List[Tuple[str, float]]:
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
    Eng-Eng monolingual frutifier.
    """

    def __init__(self, bertMLM: BertForMaskedLM, k: int):
        super().__init__(k)
        self.bertMLM = bertMLM  # this is the only layer we need.

    @staticmethod
    def frutify(desc: str, tokenizer: BertTokenizer) -> List[Tuple[str, float]]:
        pass

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (N, L) int tensor
        :return: (N, K, |S|); (num samples, k, the size of the vocabulary of subwords)
        """
        # TODO: use bertMLM.bert(), bertMLM.cls()
        pass

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        :param batch: (N, L) int tensor
        :param batch_idx: the index of the batch
        :return: (1,); the loss for this batch
        """
        # TODO: use F.cross_entropy()
        pass


class CrossLingFruit(Frutifier):
    """
    Kor-Eng cross-lingual frutifier.
    """
    def __init__(self, bert: BertModel,  k: int):
        super().__init__(k)
        # may want to add language embedding here...
        self.bert = bert
        # TODO: we need adding more layers here...

    @staticmethod
    def frutify(desc: str, tokenizer: BertTokenizer) -> List[Tuple[str, float]]:
        pass

    def forward(self, X: Tensor) -> Tensor:
        # TODO
        pass

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # TODO
        pass
