from transformers import BertTokenizer
from fruitify.configs import BERT_MODEL, K
from fruitify.datasets import Fruit2DefDataset
from fruitify.utils import load_fruit2def


def main():
    fruit2def = load_fruit2def()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    dataset = Fruit2DefDataset(fruit2def, tokenizer, K)
    for sample in dataset:
        x, y = sample
        print(x.shape)
        print(y.item())


if __name__ == '__main__':
    main()
