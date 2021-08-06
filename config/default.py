from yacs.config import CfgNode as Node
import os

cfg = Node()
cfg.OUTPUT_DIR = '/data/yhy/project/ecg_generation/output'
cfg.DESC = 'u-net'

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
cfg.DATA = Node()
cfg.DATA.DATA_ROOT = '/data/yhy/project/ecg_generation/dataset/interval'
cfg.DATA.anno_dir = '/data/yhy/project/ecg_generation/output/tianchi_interval'
cfg.DATA.train_json_path = '/data/yhy/project/ecg_generation/dataset/tianchi_train_jsons.txt'
cfg.DATA.test_json_path = '/data/yhy/project/ecg_generation/dataset/tianchi_test_jsons.txt'
cfg.DATA.CROP_SIZE = 1280
cfg.DATA.K_FOLD = True
cfg.DATA.K_TOTAL = 5
cfg.DATA.LEAD = -1


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
cfg.MODEL = Node()
cfg.MODEL.MODEL = 'u-net'
cfg.MODEL.LOSS = 'dice'


