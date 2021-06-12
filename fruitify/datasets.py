"""
for pre-processing data
"""
from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import BertTokenizer
from fruitify.configs import CLASSES


class Fruit2DefDataset(Dataset):
    @staticmethod
    def build_X(**args):
        raise NotImplementedError

    @staticmethod
    def build_y(**args):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError


class MonoFruit2DefDataset(Fruit2DefDataset):
    """
    This structure follows
    """
    # should I change this into ... encodings & labels?
    def __init__(self,
                 fruit2def: List[Tuple[str, str]],
                 tokenizer: BertTokenizer,
                 k: int):
        # (N, 3, L)
        self.X = self.build_X([def_ for _, def_ in fruit2def], tokenizer, k)
        # (N,)
        self.y = self.build_y([fruit for fruit, _ in fruit2def], CLASSES)

    @staticmethod
    def build_X(defs: List[str], tokenizer: BertTokenizer, k: int) -> Tensor:
        lefts = [" ".join(["[MASK]"] * k)] * len(defs)
        rights = defs
        encodings = tokenizer(text=lefts,
                              text_pair=rights,
                              add_special_tokens=True,
                              return_tensors="pt",
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


class UnalignedCrossFruit2DefDataset(Fruit2DefDataset):
    """
    to be used for training an UnalignedCrossFruit model.
    """

    def __init__(self):
        pass

    @staticmethod
    def build_X(**args):
        pass

    @staticmethod
    def build_y(**args):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
