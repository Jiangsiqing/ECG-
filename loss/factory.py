from .bce_loss import BceLoss
from .dice_loss import MulticlassDiceLoss
from .loss_wrapper import Loss_Wrapper


def get_loss_func(cfg):
    loss_f = cfg.MODEL.LOSS
    if loss_f == 'dice':
        return MulticlassDiceLoss()
    elif loss_f == 'bce':
        return BceLoss()
    elif loss_f == 'bce+dice':
        return Loss_Wrapper()
