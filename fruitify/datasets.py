"""
for pre-processing data
"""
from typing import List, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Fruit2DefDataset(Dataset):
    def __init__(self,
                 fruit2def: List[Tuple[str]],
                 k: int,
                 tokenizer: BertTokenizer):
        # just make sure they are of the same size.
        # TODO: preprocess to build (N, L) /  (N, |S|)
        self.X: Tensor = ...  #
        self.Y: Tensor = ...  # one-hot vector.

    @staticmethod
    def build_X(defs: List[str]) -> Tensor:
        pass

    @staticmethod
    def build_Y(fruits: List[str]) -> Tensor:
        pass

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

