"""
test for dimensions
"""
import unittest
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from fruitify.configs import BERT_MODEL
from fruitify.datasets import Fruit2DefDataset
from fruitify.loaders import load_fruit2def
from fruitify.models import MonoLingRD, UnalignedCrossLingRD, UnalignedCrossLingRDBertModel
from fruitify.vocab import build_word2subs


class TestMonoLingRD(unittest.TestCase):

    mono_rd: MonoLingRD
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
        word2subs = build_word2subs(tokenizer, k=k)
        cls.mono_rd = MonoLingRD(bert_mlm, word2subs, k=k, lr=lr)
        cls.S = tokenizer.vocab_size
        # load a single batch
        fruit2def = load_fruit2def()
        dataset = Fruit2DefDataset(fruit2def, tokenizer, k=k)
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


class TestUnalignedCrossLingRD(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # set up the mono rd
        k = 10
        batch_size = 10
        lr = 0.001
        num_langs = 2
        bert_ucl = UnalignedCrossLingRDBertModel.from_pretrained(BERT_MODEL, num_langs=num_langs)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        word2subs = build_word2subs(tokenizer, k=k)
        cls.cross_rd = UnalignedCrossLingRD(bert_ucl, word2subs, k=k, lr=lr)
        cls.S = tokenizer.vocab_size
        # load a single batch
        fruit2def = load_fruit2def()
        dataset = Fruit2DefDataset(fruit2def, tokenizer, k=k)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in loader:
            X, y = batch
            cls.X = X
            cls.y = y
            break
        cls.V = word2subs.shape[0]

    def test_forward_dim(self):
        # (N, 3, L) -> (N, K, |S|)
        S_subword = self.cross_rd.forward(self.X)
        self.assertEqual(S_subword.shape[0], self.X.shape[0])
        self.assertEqual(S_subword.shape[1], self.cross_rd.hparams['k'])
        self.assertEqual(S_subword.shape[2], self.S)

    def test_S_word_dim(self):
        # (N, 3, L) -> (N, K, |S|)
        S_subword = self.cross_rd.forward(self.X)
        # (N, K, |S|) -> (N, |V|)
        S_word = self.cross_rd.S_word(S_subword)
        self.assertEqual(S_word.shape[0], self.X.shape[0])
        self.assertEqual(S_word.shape[1], self.V)

    def test_training_step_dim(self):
        # (N, 3, L) -> scalar
        loss = self.cross_rd.training_step((self.X, self.y), 0)
        self.assertEqual(len(loss.shape), 0)  # should be a scalar
