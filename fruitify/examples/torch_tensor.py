from torch import Tensor


def main():
    x = Tensor([[1, 2, 3, 4]])
    x = x.sum()
    print(len(x.shape))


if __name__ == '__main__':
    main()
