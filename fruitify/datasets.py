"""
for pre-processing data
"""
from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Fruit2DefDataset(Dataset):
    """
    This structure follows
    """
    # these are the fruits
    CLASSES = ['apple', 'banana', 'strawberry', 'orange', 'grape']

    # should I change this into ... encodings & labels?
    def __init__(self,
                 fruit2def: List[Tuple[str, str]],
                 tokenizer: BertTokenizer,
                 k: int):
        # (N, 3, L)
        self.X = self.build_encodings([def_ for _, def_ in fruit2def], tokenizer, k)
        # (N,)
        self.y = self.build_labels([fruit for fruit, _ in fruit2def], self.CLASSES)

    @staticmethod
    def build_encodings(defs: List[str], tokenizer: BertTokenizer, k: int) -> Tensor:
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
    def build_labels(fruits: List[str], classes: List[str]) -> Tensor:
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
