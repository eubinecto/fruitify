from torch.utils.data import DataLoader
from transformers import BertTokenizer
from fruitify.configs import BERT_MODEL
from fruitify.datasets import Fruit2DefDataset
from fruitify.loaders import load_fruit2def


def main():
    fruit2def = load_fruit2def()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    dataset = Fruit2DefDataset(fruit2def, tokenizer, K)
    print("--- a sample ---")
    for sample in dataset:
        print(sample[0])
        print(sample[0].shape)
        print(sample[1])
        print(sample[1].shape)
        break

    print("--- a batch ---")
    loader = DataLoader(dataset, batch_size=10,
                        shuffle=False)
    for batch in loader:
        X, y = batch
        print(X.shape)
        print(y.shape)
        break


if __name__ == '__main__':
    main()
