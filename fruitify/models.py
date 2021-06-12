"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from typing import Tuple, List, Optional
import pytorch_lightning as pl
import torch
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM, BertEmbeddings
from transformers.models.bert.configuration_bert import BertConfig
from torch.nn import functional as F
from fruitify.configs import CLASSES


class Frutifier(pl.LightningModule):
    # fruitify into these!
    def __init__(self, k: int):
        super().__init__()
        # -- hyper params --- #
        self.k = k

    def frutify(self, desc: str, tokenizer: BertTokenizer) -> List[Tuple[str, float]]:
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

    def frutify(self, desc: str, tokenizer: BertTokenizer) -> List[Tuple[str, float]]:
        # TODO: Get use of fruitify.configs.CLASSES
        pass

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (N, 3, L) (num samples, 0=input_ids/1=token_type_ids/2=attention_mask, the maximum length)
        :return: (N, K, |S|); (num samples, k, the size of the vocabulary of subwords)
        """
        # TODO: Get use of bert_mlm.bert(), bert_mlm.cls(). Help: examples/bert_mlm.py
        H_k = ...
        S_subword = ...
        return S_subword

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        :param batch: A tuple of X and Y; ((N, 3, L), (N,)).
        :param batch_idx: the index of the batch
        :return: (1,); the loss for this batch
        """
        X, y = batch
        S_subword = self.forward(X)
        loss_ = F.cross_entropy(S_subword, y)
        loss = loss_.sum()
        return loss


class UnalignedCrossLingBertEmbeddings(BertEmbeddings):
    def __init__(self, num_langs: int,  config: BertConfig):
        super(UnalignedCrossLingBertEmbeddings, self).__init__(config)
        self.num_Langs = num_langs
        self.config = config
        self.lang_ids: Optional[Tensor] = None  # to be updated
        # additional learnable embeddings at the bottom
        self.language_embedding = torch.nn.Embedding(num_embeddings=self.num_Langs,
                                                     # match with the embeddings.
                                                     embedding_dim=self.hidden_size)
        # initialise the weights of the language embedding
        torch.nn.init.uniform_(self.language_embedding.weight.data,
                               a=-np.sqrt(3 / self.language_embedding.weight.data.shape[1]),
                               b=np.sqrt(3 / self.language_embedding.weight.data.shape[1]))

    def forward(self, *args) -> Tensor:
        out = super(UnalignedCrossLingBertEmbeddings, self).forward(*args)
        # just add the embeddings at the end.
        return out + self.language_embedding(self.lang_ids)

    # just a clever way of passing the lang_ids
    def update_lang_ids(self, lang_ids: Tensor):
        self.lang_ids = lang_ids

    def del_lang_ids(self):
        del self.lang_ids
        self.lang_ids = None


# yeah, it would be better to subclass BERT here.
class UnalignedCrossLingBertModel(BertModel):

    def __init__(self, config: BertConfig, num_langs: int):
        super().__init__(config)
        self.embeddings = UnalignedCrossLingBertEmbeddings(num_langs, config)

    def forward(self,
                # this one is added
                lang_ids: Tensor = None,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,):
        # first, register language ids, then call forward.
        self.embeddings.update_lang_ids(lang_ids)
        H_all = super(UnalignedCrossLingBertModel, self).forward(input_ids, attention_mask,
                                                                 token_type_ids, position_ids,
                                                                 head_mask, inputs_embeds, encoder_hidden_states,
                                                                 encoder_attention_mask, past_key_values,
                                                                 use_cache, output_attentions, output_hidden_states,
                                                                 return_dict)
        self.embeddings.del_lang_ids()  # delete it after use
        return H_all

    def _reorder_cache(self, past, beam_idx):
        super(UnalignedCrossLingBertModel, self)._reorder_cache(past, beam_idx)


class UnalignedCrossLingFruit(Frutifier):
    """
    Kor-Eng unaligned cross-lingual fruitifier.
    """
    def __init__(self, bert_ucl: UnalignedCrossLingBertModel, k: int):
        if isinstance(bert_ucl.config, BertConfig):
            super().__init__(k)
            # may want to add language embedding here...
            self.bert_ucl = bert_ucl  # we are using multi-lingual bert for this.

    def frutify(self, desc: str, tokenizer: BertTokenizer) -> List[Tuple[str, float]]:
        # TODO: Get use of cls.FRUITS
        pass

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (N, 4, L) (num samples, 0=input_ids/1=token_type_ids/2=attention_mask/3=lang_ids, the maximum length)
        :return: S_subword; (N, K, |S|).
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        lang_ids = X[:, 3]
        H_all = self.bert_ucl.forward(lang_ids, input_ids, attention_mask, token_type_ids)
        H_k = H_all[1:self.k + 1]  # (N, K, 768)
        Emb_token = self.bert_ucl.embeddings.word_embeddings.weight.data  # (N, |S|, 768)
        S_subword = torch.einsum('nkh,nsh->nks', H_k, Emb_token)  # ( N, K, 768) * (N, |S|, 768) -> (N, K, |S|)
        return S_subword

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        X, y_1, y_2 = batch  # the encodings, labels_fruit, labels_lang
        # TODO: use F.cross_entropy to compute the loss, and return it
        # you should group them by y_2.
        S_subword = self.forward(X)
        loss = ...
        return loss
