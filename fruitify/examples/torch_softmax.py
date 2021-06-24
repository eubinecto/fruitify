
import torch
from torch.nn import functional as F


def main():
    X = torch.randn(size=(2, 3, 10))

    X = F.softmax(X, dim=2)
    print(X[0, 0])


if __name__ == '__main__':
    main()