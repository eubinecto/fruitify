from torch import Tensor


def main():
    x = Tensor([[1, 2], [3, 4]])
    print(type(x.size()[1]))


if __name__ == '__main__':
    main()