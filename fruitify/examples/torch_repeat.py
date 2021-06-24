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

    # from 2d to 3d
    X_2 = torch.Tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print(X_2.repeat(5, 1, 1).shape)

    # 3d tensor
    x_3 = torch.rand(size=(10, 59, 768))
    print(x_3.repeat(3, 1, 1).shape)


if __name__ == '__main__':
    main()
