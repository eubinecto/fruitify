from torch.utils.data import DataLoader
from transformers import BertTokenizer
from fruitify.configs import BERT_MODEL, MBERT_MODEL
from fruitify.datasets import MonoENFruit2Def, CrossFruit2Def
from fruitify.loaders import load_fruitify_dataset


def main():
    k = 3  # three is enough for this.
    fruitify_dataset = load_fruitify_dataset()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    mono_en_dataset = MonoENFruit2Def(fruitify_dataset, tokenizer, k)
    print("--- a sample ---")
    for sample in mono_en_dataset:
        X, y = sample
        print(X)
        input_ids = X[0]
        decoded = [tokenizer.decode(token_id) for token_id in input_ids.tolist()]
        print(decoded)
        print(X.shape)  # (3, L)
        print(y)  # just a scalar tensor
        print(y.shape)
        break

    print("--- a batch ---")
    loader = DataLoader(mono_en_dataset, batch_size=10,
                        shuffle=False)
    for batch in loader:
        X, y = batch
        print(X)
        print(X.shape)  # (N, 3, L)
        print(y)  # (N,)
        print(y.shape)
        break

    print("---cross---")
    k = 10  # should be long enough.
    tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL)
    cross_dataset = CrossFruit2Def(fruitify_dataset, tokenizer, k)

    print("--- a sample ---")
    for sample in cross_dataset:
        X, y = sample
        print(X)
        input_ids = X[0]
        decoded = [tokenizer.decode(token_id) for token_id in input_ids.tolist()]
        print(decoded)
        print(X.shape)  # (3, L)
        print(y)  # just a scalar tensor
        print(y.shape)
        break

    print("--- a batch ---")
    loader = DataLoader(cross_dataset, batch_size=10,
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
