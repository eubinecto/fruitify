"""
for pre-processing data
"""
from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer
from fruitify.vocab import LANGS


class Fruit2DefDataset(Dataset):
    def __init__(self,
                 fruit2def: List[Tuple[str, str, str]],
                 tokenizer: BertTokenizer,
                 k: int):
        self.X = self.build_X(fruit2def, tokenizer, k)
        self.y = self.build_y(fruit2def, tokenizer, k)

    @staticmethod
    def build_X(fruit2def: List[Tuple[str, str, str]], tokenizer: BertTokenizer, k: int) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def build_y(fruit2def: List[Tuple[str, str, str]], classes: List[str]) -> Tensor:
        raise NotImplementedError

    def upsample(self, repeat: int):
        """
        this is to try upsampling the batch by simply repeating the instances.
        https://github.com/eubinecto/fruitify/issues/7#issuecomment-860603350
        :return:
        """
        self.X = self.X.repeat(repeat, 1, 1)  # (N, 3, L) -> (N * repeat, 3, L)
        self.y = self.y.repeat(repeat)  # (N,) -> (N * repeat, )

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx], self.y[idx]


class MonoFruit2Def(Fruit2DefDataset):
    """
    This structure follows
    """

    @staticmethod
    def build_X(fruit2def: List[Tuple[str, str, str]], tokenizer: BertTokenizer, k: int) -> Tensor:
        defs = [def_ for _, _, def_ in fruit2def]
        lefts = [" ".join(["[MASK]"] * k)] * len(defs)
        rights = defs
        encodings = tokenizer(text=lefts,
                              text_pair=rights,
                              return_tensors="pt",
                              add_special_tokens=True,
                              truncation=True,
                              padding=True)

        return torch.stack([encodings['input_ids'],
                            # token type for the padded tokens? -> they are masked with the
                            # attention mask anyways
                            # https://github.com/google-research/bert/issues/635#issuecomment-491485521
                            encodings['token_type_ids'],
                            encodings['attention_mask']], dim=1)

    @staticmethod
    def build_y(fruits: List[str], classes: List[str]) -> Tensor:
        return Tensor([
            classes.index(fruit)
            for fruit in fruits
        ]).long()

    def upsample(self, repeat: int):
        """
        this is to try upsampling the batch by simply repeating the instances.
        https://github.com/eubinecto/fruitify/issues/7#issuecomment-860603350
        :return:
        """
        self.X = self.X.repeat(repeat, 1, 1)  # (N, 3, L) -> (N * repeat, 3, L)
        self.y = self.y.repeat(repeat)  # (N,) -> (N * repeat, )

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx], self.y[idx]

