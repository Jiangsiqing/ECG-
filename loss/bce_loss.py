from torch import nn
import torch
from torch.nn import functional as F


class BceLoss(nn.Module):
    def __init__(self):
        super(BceLoss, self).__init__()

    def forward(self, logits, gt):
        bce_loss = F.binary_cross_entropy(logits.float(), gt.float())
        # bce_loss = F.binary_cross_entropy_with_logits(logits.float(), gt.float())
        return bce_loss
