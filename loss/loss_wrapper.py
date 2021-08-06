from torch import nn
import torch
from torch.nn import functional as F
from .dice_loss import MulticlassDiceLoss
from .bce_loss import BceLoss


class Loss_Wrapper(nn.Module):
    def __init__(self):
        super(Loss_Wrapper, self).__init__()

    def forward(self, logits, gt):
        bce_loss = BceLoss()(logits, gt)
        dice_loss = MulticlassDiceLoss()(logits, gt)
        return bce_loss + dice_loss
