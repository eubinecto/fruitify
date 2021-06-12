"""
https://pytorch.org/docs/stable/generated/torch.gather.html
gather values along an axis specified by dim.

out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
"""

import torch


def main():
    X_1 = torch.tensor([[1, 2],
                        [3, 4],
                        [5, 6]])
    X_2 = torch.tensor([[0, 0],
                        [1, 1],
                        [1, 0]])
    # both index and the input should be of the same size.
    print(torch.gather(input=X_1, dim=0, index=X_2))  # vertical indexing
    print(torch.gather(input=X_1, dim=1, index=X_2))  # horizontal indexing
    print(torch.gather(input=X_1, dim=-1, index=X_2))  # horizontal indexing


if __name__ == '__main__':
    main()