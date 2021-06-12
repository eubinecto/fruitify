from torch import Tensor


def main():
    X_1 = Tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8]])  # (2, 4)
    x_2 = Tensor([[0, 3],  # 0
                  [1, 2]]).long()  # (2, 2)
    Y = X_1[:, x_2]  # choose everything on the first dim, but take only x_2 indices for the second dim
    print(Y)
    print(Y.shape)


if __name__ == '__main__':
    main()
