"""
https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
"""
import torch


def main():
    # an Embedding module containing 10 tensors of size 3
    embedding = torch.nn.Embedding(10, 3)
    # a batch of 2 samples of 4 indices each
    X = torch.LongTensor([[1, 2, 4, 5],
                          [4, 3, 2, 9]])  # (2, 4)
    # retrieve their embeddings using indices
    Y = embedding(X)  # (2, 4) -> (2, 4, 3)
    print(Y)
    print(Y.shape)


if __name__ == '__main__':
    main()
