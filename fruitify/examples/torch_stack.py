
import torch


def main():
    x_1 = torch.Tensor([[1, 2, 3, 4],
                        [5, 6, 7, 8]])  # (2, 4)
    x_2 = torch.Tensor([[1, 2, 3, 4],
                        [5, 6, 7, 8]])  # (2, 4)
    x_3 = torch.Tensor([[1, 2, 3, 4],
                        [5, 6, 7, 8]])  # (2, 4)
    # how do I stack them into (N=2, 3, 4)?
    staked = torch.stack([x_1, x_2, x_3], dim=1)
    print(staked.shape)


if __name__ == '__main__':
    main()
