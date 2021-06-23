"""
https://seducinghyeok.tistory.com/9
"""
import torch


def main():
    # 1d tensor
    x_1 = torch.tensor([1, 2, 3])
    # repeat 3 times in dim=0 (increases the dimension in rows)
    # repeat 2 times in dim=1 (increases the dimension in columns)
    print(x_1.repeat(3, 2))

    # 3d tensor
    x_3 = torch.ones(size=(10, 59, 768))
    print(x_3.repeat(3, 1, 1).shape)


if __name__ == '__main__':
    main()
