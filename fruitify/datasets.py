"""
for pre-processing data
"""
from typing import List
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Fruit2DefDataset(Dataset):
    def __init__(self,
                 fruits: List[str],
                 defs: List[str],
                 k: int,
                 tokenizer: BertTokenizer):
        # just make sure they are of the same size.
        assert len(fruits) == len(defs)
        # TODO: preprocess to build (N, L)
        self.X: Tensor = ...

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx]
