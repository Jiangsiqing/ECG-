import torch
import numpy as np


def to_one_hot(mask, n_class):
    """
    Transform a mask to one hot
    change a mask to n * h* w   n is the class
    Args:
        mask:
        n_class: number of class for segmentation
    Returns:
        y_one_hot: one hot mask
    """
    y_one_hot = torch.zeros((mask.shape[0], n_class, mask.shape[2]))
    y_one_hot = y_one_hot.scatter(1, mask, 1).long()
    return y_one_hot


if __name__ == '__main__':

    x = torch.LongTensor(16, 1, 1280).random_() % 4
    print(x)
    mask = to_one_hot(x, 4)
    print(mask)
    print(mask.shape)