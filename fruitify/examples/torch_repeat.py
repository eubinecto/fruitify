"""
https://seducinghyeok.tistory.com/9
"""


import torch


def main():
    x = torch.tensor([1, 2, 3])
    # repeat 3 times in dim=0 (increases the dimension in rows)
    # repeat 2 times in dim=1 (increases the dimension in columns)
    print(x.repeat(3, 3))


if __name__ == '__main__':
    main()
