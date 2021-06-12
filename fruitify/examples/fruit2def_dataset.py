from torch.utils.data import DataLoader
from transformers import BertTokenizer
from fruitify.configs import BERT_MODEL
from fruitify.datasets import MonoFruit2DefDataset
from fruitify.loaders import load_fruit2def


def main():
    fruit2def = load_fruit2def()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    dataset = MonoFruit2DefDataset(fruit2def, tokenizer, 5)
    print("--- a sample ---")
    for sample in dataset:
        X, y = sample
        print(X)
        print(X.shape)  # (3, L)
        print(y)  # just a scalar tensor
        print(y.shape)
        break

    print("--- a batch ---")
    loader = DataLoader(dataset, batch_size=10,
                        shuffle=False)
    for batch in loader:
        X, y = batch
        print(X)
        print(X.shape)  # (N, 3, L)
        print(y)  # (N,)
        print(y.shape)
        break


if __name__ == '__main__':
    main()
