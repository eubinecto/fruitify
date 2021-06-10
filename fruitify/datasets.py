"""
for pre-processing data
"""
from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Fruit2DefDataset(Dataset):
    # this is the label.
    CLASSES = ['apple', 'banana', 'strawberry', 'orange', 'grape']

    # should I change this into ... encodings & labels?
    def __init__(self,
                 fruit2def: List[Tuple[str]],
                 tokenizer: BertTokenizer,
                 k: int):
        # just make sure they are of the same size.
        # TODO: preprocess to build (N, 3, L) /  (N,)
        self.X: Tensor = self.build_X([_def for _, _def in fruit2def], tokenizer, k)  # (N, 3, L)
        self.Y: Tensor = self.build_Y([fruit for fruit, _ in fruit2def], self.CLASSES)  # (N,)

    @staticmethod
    def build_X(defs: List[str], tokenizer: BertTokenizer, k: int) -> Tensor:
        # encode the definitions into the inputs for RD task.
        inputs = [
            " ".join(["[CLS]"] + (["[MASK]"] * k)) + "[SEP]" + _def
            for _def in defs
        ]
        encoded = tokenizer(inputs, return_tensors="pt", padding=True)
        input_ids = encoded['input_ids']  # (N, L)
        token_type_ids = encoded['token_type_ids']  # (N, L)
        attention_mask = encoded['attention_mask']  # (N, L)
        return torch.stack([input_ids,
                            token_type_ids,
                            attention_mask], dim=1)  # (N, 3, L)

    @staticmethod
    def build_Y(fruits: List[str], classes: List[str]) -> Tensor:
        return Tensor([
              classes.index(fruit)
              for fruit in fruits
        ]).long()  # targets should be typed long

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx], self.Y[idx]

