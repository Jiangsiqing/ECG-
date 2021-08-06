from .unet import Unet_1d


def get_model(cfg):
    model_name = cfg.MODEL.MODEL
    model = None
    if model_name == 'u-net':
        in_channels = 8 if cfg.DATA.LEAD == -1 else 1
        model = Unet_1d(in_channels, 4)
    return model