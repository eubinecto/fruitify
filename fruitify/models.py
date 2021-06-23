"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
import os
from argparse import Namespace
from typing import Tuple, List, Optional, Union
import pytorch_lightning as pl
import torch
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM, BertEmbeddings
from transformers.models.bert.configuration_bert import BertConfig
from torch.nn import functional as F
from fruitify.datasets import Fruit2DefDataset


class RD(pl.LightningModule):
    """
    A reverse-dictionary.
    """
    # fruitify into these!
    def __init__(self, word2subs: Tensor, k: int, lr: float):
        super().__init__()
        # -- to be used to compute S_word -- #
        self.word2subs = word2subs.cuda()
        # -- hyper params --- #
        # should be saved to self.hparams
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4390#issue-730493746
        self.save_hyperparameters(Namespace(k=k, lr=lr))

    def forward(self, X: Tensor) -> Tensor:
        """
        forward defines the prediction/inference actions.
        """
        raise NotImplementedError

    def S_word(self, S_subword: Tensor) -> Tensor:
        # pineapple -> pine, ###apple, mask, mask, mask, mask, mask
        # [ ...,
        #   ...,
        #   ...
        #   [98, 122, 103, 103]]
        # [
        word2subs = self.word2subs.T.repeat(S_subword.shape[0], 1, 1)  # (|V|, K) -> (N, K, |V|)
        S_word = S_subword.gather(dim=-1, index=word2subs)  # (N, K, |S|) -> (N, K, |V|)
        S_word = S_word.sum(dim=1)  # (N, K, |V|) -> (N, |V|)
        return S_word

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        :param batch: A tuple of X, y, subword_ids; ((N, 3, L), (N,),
        :param batch_idx: the index of the batch
        :return: (1,); the loss for this batch
        """
        X, y = batch
        X = X.cuda()
        y = y.cuda()
        S_subword = self.forward(X)  # (N, 3, L) -> (N, K, |S|)
        S_word = self.S_word(S_subword)  # (N, K, |S|) -> (N, |V|)
        loss = F.cross_entropy(S_word, y)  # (N, |V|) -> (N,)
        loss = loss.sum()  # (N,) -> scalar
        self.log('train_loss', loss.item())  # monitor this, while training.
        return loss

    def on_epoch_end(self) -> None:
        pass

    def configure_optimizers(self) -> Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


class MonoLingRD(RD):
    """
    Eng-Eng monolingual fruitifier.
    """

    def __init__(self, bert_mlm: BertForMaskedLM, word2subs: Tensor, k: int = None, lr: float = None):
        super().__init__(word2subs, k, lr)
        self.bert_mlm = bert_mlm  # this is the only layer we need, as far as MonoLing RD is concerned

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (N, 3, L) (num samples, 0=input_ids/1=token_type_ids/2=attention_mask, the maximum length)
        :return: (N, K, |V|); (num samples, k, the size of the vocabulary of subwords)
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert_mlm.bert.forward(input_ids, attention_mask, token_type_ids)[0]  # (N, 3, L) -> (N, L, 768)
        H_k = H_all[:, 1: self.hparams['k'] + 1]  # (N, L, 768) -> (N, K, 768)
        S_subword = self.bert_mlm.cls(H_k)  # (N, K, 768) ->  (N, K, |S|)
        return S_subword


class UnalignedCrossLingRDBertEmbeddings(BertEmbeddings):
    def __init__(self, num_langs: int,  config: BertConfig):
        super(UnalignedCrossLingRDBertEmbeddings, self).__init__(config)
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
        out = super(UnalignedCrossLingRDBertEmbeddings, self).forward(*args)
        # just add the embeddings at the end.
        return out + self.language_embedding(self.lang_ids)

    # just a clever way of passing the lang_ids
    def update_lang_ids(self, lang_ids: Tensor):
        self.lang_ids = lang_ids

    def del_lang_ids(self):
        del self.lang_ids
        self.lang_ids = None


# yeah, it would be better to subclass BERT here.
class UnalignedCrossLingRDBertModel(BertModel):

    def __init__(self, config: BertConfig, num_langs: int):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], num_langs: int = None,
                        *model_args, **kwargs):
        model = super(UnalignedCrossLingRDBertModel, cls).from_pretrained(pretrained_model_name_or_path,
                                                                          *model_args, **kwargs)
        # override the embeddings with...hmmmmmm.....
        model.embeddings = UnalignedCrossLingRDBertEmbeddings(num_langs, model.config)

    def forward(self,
                # this one is added
                lang_ids: Tensor = None,  # the language mask.
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
        assert not self.embeddings.lang_ids  # should have been deleted
        self.embeddings.update_lang_ids(lang_ids)
        H_all = super(UnalignedCrossLingRDBertModel, self).forward(input_ids, attention_mask,
                                                                   token_type_ids, position_ids,
                                                                   head_mask, inputs_embeds, encoder_hidden_states,
                                                                   encoder_attention_mask, past_key_values,
                                                                   use_cache, output_attentions, output_hidden_states,
                                                                   return_dict)
        self.embeddings.del_lang_ids()  # delete it after use
        return H_all

    def _reorder_cache(self, past, beam_idx):
        super(UnalignedCrossLingRDBertModel, self)._reorder_cache(past, beam_idx)


class UnalignedCrossLingRD(RD):
    """
    Kor-Eng unaligned cross-lingual fruitifier.
    """
    def __init__(self, bert_ucl: UnalignedCrossLingRDBertModel, word2subs: Tensor,  k: int, lr: float):
        if isinstance(bert_ucl.config, BertConfig):
            super().__init__(word2subs, k, lr)
            # may want to add language embedding here...
            self.bert_ucl = bert_ucl  # we are using multi-lingual bert for this.

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (N, 4, L) (num samples, 0=input_ids/1=token_type_ids/2=attention_mask/3=lang_ids, the maximum length)
        :return: S_subword; (N, K, |S|).
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        lang_ids = X[:, 3]
        H_all = self.bert_ucl.forward(lang_ids, input_ids, attention_mask, token_type_ids)[0]
        H_k = H_all[:, 1:self.hparams['k'] + 1]  # (N, K, 768)
        Emb_token = self.bert_ucl.embeddings.word_embeddings.weight.data  # (N, |S|, 768)
        S_subword = torch.einsum('nkh,nsh->nks', H_k, Emb_token)  # ( N, K, 768) * (N, |S|, 768) -> (N, K, |S|)
        return S_subword


class Fruitifier:

    def __init__(self, rd: RD, tokenizer: BertTokenizer, fruits: List[str]):
        self.rd = rd  # a trained reverse dictionary
        self.tokenizer = tokenizer
        self.fruits = fruits  # the classes

    def fruitify(self, descriptions: List[str]) -> List[List[Tuple[str, float]]]:
        # get the X
        X = Fruit2DefDataset.build_X(defs=descriptions, tokenizer=self.tokenizer, k=self.rd.hparams['k']).cuda()        
        # get S_subword for this.
        S_subword = self.rd.forward(X)
        S_word = self.rd.S_word(S_subword)  # (N, |V|).
        S_word_probs = F.softmax(S_word, dim=1)  # (N, |V|) -> (N, |V|) softmax along |V|
        results = list()
        for w_probs in S_word_probs.tolist():
            fruit2score = [
                (fruit, w_score)
                for fruit, w_score in zip(self.fruits, w_probs)
            ]
            # sort and append
            results.append(sorted(fruit2score, key=lambda x: x[1], reverse=True))
        return results

