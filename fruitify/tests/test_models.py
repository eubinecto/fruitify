"""
test for dimensions
"""
import unittest
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from fruitify.configs import BERT_MODEL
from fruitify.datasets import MonoENFruit2Def
from fruitify.loaders import load_fruitify_dataset
from fruitify.models import ReverseDict
from fruitify.vocab import build_word2subs


class TestMonoLingRD(unittest.TestCase):
    mono_rd: ReverseDict
    X: Tensor
    y: Tensor
    S: int
    V: int

    @classmethod
    def setUpClass(cls) -> None:
        # set up the mono rd
        k = 3
        batch_size = 10
        lr = 0.001
        bert_mlm = BertForMaskedLM.from_pretrained(BERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        word2subs = build_word2subs(tokenizer, k=k, mode="mono_en")
        cls.mono_rd = ReverseDict(bert_mlm, word2subs, k=k, lr=lr)
        cls.S = tokenizer.vocab_size
        # load a single batch
        dataset = MonoENFruit2Def(load_fruitify_dataset(), tokenizer, k=k)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in loader:
            X, y = batch
            cls.X = X
            cls.y = y
            break

        cls.V = word2subs.shape[0]

    def test_forward_dim(self):
        # (N, 3, L) -> (N, K, |S|)
        S_subword = self.mono_rd.forward(self.X)
        self.assertEqual(S_subword.shape[0], self.X.shape[0])
        self.assertEqual(S_subword.shape[1], self.mono_rd.hparams['k'])
        self.assertEqual(S_subword.shape[2], self.S)

    def test_S_word_dim(self):
        # (N, 3, L) -> (N, K, |S|)
        S_subword = self.mono_rd.forward(self.X)
        # (N, K, |S|) -> (N, |V|)
        S_word = self.mono_rd.S_word(S_subword)
        self.assertEqual(S_word.shape[0], self.X.shape[0])
        self.assertEqual(S_word.shape[1], self.V)

    def test_training_step_dim(self):
        # (N, 3, L) -> scalar
        loss = self.mono_rd.training_step((self.X, self.y), 0)
        self.assertEqual(len(loss.shape), 0)  # should be a scalar
