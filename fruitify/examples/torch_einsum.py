
import torch
from torch import Tensor
import numpy as np


def main():
    N = 10
    K = 5
    H = 768
    S = 10000
    X_1 = Tensor(np.ones(shape=(N, K, H)))  # N, K, H
    X_2 = Tensor(np.ones(shape=(N, S, H)))  # N, S, H

    # How do I get  (N, K, S)? use einsum
    # https://discuss.pytorch.org/t/3d-matrix-of-dot-products/89198
    Y = torch.einsum('nkh,nsh->nks', X_1, X_2)  # (N, K, H) * (N, S, H) -> (N, K, S)
    print(Y.shape)
    # einsum is awesome!


if __name__ == '__main__':
    main()
