"""
https://pytorch.org/docs/stable/generated/torch.argsort.html

"""
import torch


def main():
    # (N, |V|)
    X = torch.tensor([[1, 4, 3, 2],
                      [5, 8, 7, 6]])
    print(X.shape)
    print(torch.argsort(X, dim=0))
    print(torch.argsort(X, dim=1))
    print(torch.argsort(X, dim=1, descending=True))


if __name__ == '__main__':
    main()
