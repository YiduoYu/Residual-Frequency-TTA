"""Default configuration for the ReSpect-TTA experiment project."""

import math
from yacs.config import CfgNode as CN


_C = CN()
# random seed number
_C.SEED = 0
# number of gpus per node
_C.NUM_GPUS = 8
_C.VISIBLE_DEVICES = 0
# directory to save result txt file
_C.RESULT_DIR = 'results/'
_C.NORMALIZE = 'NST'

_C.DATA_LOADER = CN()
_C.DATA_LOADER.NUM_WORKERS = 2
_C.DATA_LOADER.PIN_MEMORY = True
_C.DATA_LOADER.DROP_LAST = True

_C.DATA = CN()
_C.DATA.BASE_DIR = 'data/'
_C.DATA.NAME = 'weather'
_C.DATA.N_VAR = 21
_C.DATA.SEQ_LEN = 96
_C.DATA.LABEL_LEN = 48
_C.DATA.PRED_LEN = 96
_C.DATA.FEATURES = 'M'
_C.DATA.TIMEENC = 0
_C.DATA.FREQ = 'h'
_C.DATA.SCALE = "standard"  # standard, min-max
_C.DATA.TRAIN_RATIO = 0.7
_C.DATA.TEST_RATIO = 0.2
_C.DATA.DATE_IDX = 0
_C.DATA.TARGET_START_IDX = 0
_C.DATA.PERIOD_LEN = 24  # Used only when SAN is ENABLED
_C.DATA.STATION_TYPE = 'adaptive'  # Used only when SAN is ENABLED

_C.TRAIN = CN()
_C.TRAIN.ENABLE = False
_C.TRAIN.SPLIT = 'train'
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.SHUFFLE = True
_C.TRAIN.DROP_LAST = True
# directory to save checkpoints
_C.TRAIN.CHECKPOINT_DIR = 'results/'
# path to checkpoint to resume training
_C.TRAIN.RESUME = ''
# epoch period to evaluate on a validation set
_C.TRAIN.EVAL_PERIOD = 5
# iteration frequency to print progress meter
_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.BEST_METRIC_INITIAL = float("inf")
_C.TRAIN.BEST_LOWER = True

_C.VAL = CN()
_C.VAL.SPLIT = 'val'
_C.VAL.BATCH_SIZE = 256
_C.VAL.SHUFFLE = False
_C.VAL.DROP_LAST = False
_C.VAL.VIS = False

_C.TEST = CN()
_C.TEST.ENABLE = True
_C.TEST.SPLIT = 'test'
_C.TEST.BATCH_SIZE = 256
_C.TEST.SHUFFLE = False
_C.TEST.DROP_LAST = False

_C.TTA = CN()
_C.TTA.ENABLE = False
_C.TTA.MODULE_NAMES_TO_ADAPT = 'cali'  # all, norm, etc
_C.TTA.LOG = False
_C.TTA.SOLVER = CN()
_C.TTA.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.TTA.SOLVER.BASE_LR = 0.005
_C.TTA.SOLVER.WEIGHT_DECAY = 0.0001
_C.TTA.SOLVER.MOMENTUM = 0.9
_C.TTA.SOLVER.NESTEROV = True
_C.TTA.SOLVER.DAMPENING = 0.0
_C.TTA.RESPECT = CN()
_C.TTA.RESPECT.BATCH_SIZE = 48
_C.TTA.RESPECT.STEPS = 3
_C.TTA.RESPECT.BUFFER_SIZE = 10
_C.TTA.RESPECT.BUFFER_CONTEXT_SIZE = 10
_C.TTA.RESPECT.ADAPT_FREQUENCY = 50

# Online adaptation settings
_C.TTA.RESPECT.FAST_ADAPTATION = True
_C.TTA.RESPECT.ADAPTIVE_LR = True
_C.TTA.RESPECT.MAX_LR = 0.005
_C.TTA.RESPECT.MIN_LR = 0.0001
_C.TTA.RESPECT.MOMENTUM_FACTOR = 0.9
_C.TTA.RESPECT.CONVERGENCE_THRESHOLD = 1e-4
_C.TTA.RESPECT.VAR_WISE_GATING = True
_C.TTA.RESPECT.ADAPTER_LAYERS = 1
_C.TTA.RESPECT.HIDDEN_DIM = 64
_C.TTA.RESPECT.PER_BATCH_LR_RESET = True

# CSV export settings
_C.TTA.RESPECT.SAVE_CSV = False

# Residual spectral correction settings
_C.TTA.RESPECT.SPECTRAL = CN()
_C.TTA.RESPECT.SPECTRAL.MODE = 'concat'
_C.TTA.RESPECT.SPECTRAL.USAGE = 'context_plus_direct'
_C.TTA.RESPECT.SPECTRAL.SOURCE = 'residual_profile'
_C.TTA.RESPECT.SPECTRAL.REPR = 'real_imag'
_C.TTA.RESPECT.SPECTRAL.K = 8
_C.TTA.RESPECT.SPECTRAL.HISTORY_SIZE = 8
_C.TTA.RESPECT.SPECTRAL.INCLUDE_POWER_RATIO = True
_C.TTA.RESPECT.SPECTRAL.INCLUDE_RESIDUAL_STATS = True
_C.TTA.RESPECT.SPECTRAL.MIN_POWER_RATIO_FOR_DIRECT = 0.20
_C.TTA.RESPECT.SPECTRAL.USE_CONTEXT_PROJECTION = True
_C.TTA.RESPECT.SPECTRAL.CONTEXT_PROJECTION_DIM = 8
_C.TTA.RESPECT.SPECTRAL.MAX_CONTEXT_DIM_FOR_FULL_ADAPTER = 16
_C.TTA.RESPECT.SPECTRAL.PARAM_BUDGET_MULTIPLIER = 1.25
_C.TTA.RESPECT.SPECTRAL.ENABLE_DIRECT_CORRECTION = True
_C.TTA.RESPECT.SPECTRAL.CORRECTION_SOURCE = 'complex_modulation'
_C.TTA.RESPECT.SPECTRAL.GATE_MODE = 'unidirectional'
_C.TTA.RESPECT.SPECTRAL.ENABLE_REFERENCE_CONTENT_INJECTION = False
_C.TTA.RESPECT.SPECTRAL.USE_GLOBAL_GATE = True
_C.TTA.RESPECT.SPECTRAL.GLOBAL_GATE_FEATURES = 'hs,l2,low_freq_ratio'
_C.TTA.RESPECT.SPECTRAL.GLOBAL_GATE_INIT_BIAS = -2.0
_C.TTA.RESPECT.SPECTRAL.GLOBAL_GATE_LR_MULTIPLIER = 10.0
_C.TTA.RESPECT.SPECTRAL.GLOBAL_GATE_FLOOR = 0.0
_C.TTA.RESPECT.SPECTRAL.CONSTANT_GLOBAL_GATE = -1.0
_C.TTA.RESPECT.SPECTRAL.COMPLEX_MOD_FREQ_CUTOFF_RATIO = 1.0
_C.TTA.RESPECT.SPECTRAL.ENABLE_REVIN_ON_SPECTRAL_PATH = False
_C.TTA.RESPECT.SPECTRAL.WINDOW_TYPE = 'hann'
_C.TTA.RESPECT.SPECTRAL.OUT_PROJ = 'diagonal'
_C.TTA.RESPECT.SPECTRAL.FFT_PRECISION = 'fp32'
_C.TTA.RESPECT.SPECTRAL.FFT_LENGTH_MODE = 'exact'
_C.TTA.RESPECT.SPECTRAL.LOG_CSV_PATH = ''


_C.MODEL = CN()
_C.MODEL.NAME = 'iTransformer'
_C.MODEL.task_name = 'long_term_forecast'
_C.MODEL.seq_len = _C.DATA.SEQ_LEN 
_C.MODEL.label_len = _C.DATA.LABEL_LEN # Not needed in iTransformer
_C.MODEL.pred_len = _C.DATA.PRED_LEN 
_C.MODEL.e_layers = 4
_C.MODEL.d_layers = 1 # Not needed in iTransformer
_C.MODEL.factor = 3 # Not used in iTransformer Full Attention. Used in Prob Attention (probabilistic attention) in informer
_C.MODEL.enc_in = _C.DATA.N_VAR # Used only in classification
_C.MODEL.dec_in = _C.DATA.N_VAR # Not needed in iTransformer
_C.MODEL.c_out = _C.DATA.N_VAR # Not needed in iTransformer
_C.MODEL.d_model = 512 # embedding dimension
_C.MODEL.d_ff = 512  # feedforward dimension d_model -> d_ff -> d_model
_C.MODEL.moving_avg = 25
_C.MODEL.output_attention = False # whether the attention weights are returned by the forward method of the attention class
_C.MODEL.dropout = 0.1
_C.MODEL.n_heads = 8
_C.MODEL.activation = 'gelu'
_C.MODEL.channel_independence = True
_C.MODEL.METRIC_NAMES = ('MAE',)
_C.MODEL.LOSS_NAMES = ('MSE',)
_C.MODEL.embed = 'timeF'
_C.MODEL.freq = 'h'
_C.MODEL.ignore_stamp = False
# OLS params
_C.MODEL.instance_norm = True
_C.MODEL.individual = False
_C.MODEL.alpha = 0.000001

_C.NORM_MODULE = CN()
_C.NORM_MODULE.ENABLE = False  # NST
_C.NORM_MODULE.NAME = 'SAN'  # SAN, RevIN, DishTS

_C.SAN = CN()
_C.SAN.RESULT_DIR = 'results/station/'
_C.SAN.TRAIN = CN()
_C.SAN.TRAIN.CHECKPOINT_DIR = 'results/station/'
_C.SAN.SOLVER = CN()
_C.SAN.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.SAN.SOLVER.START_EPOCH = 0
_C.SAN.SOLVER.MAX_EPOCH = 10
_C.SAN.SOLVER.BASE_LR = 0.001
_C.SAN.SOLVER.WEIGHT_DECAY = 0.0001
_C.SAN.SOLVER.MOMENTUM = 0.9
_C.SAN.SOLVER.NESTEROV = True
_C.SAN.SOLVER.DAMPENING = 0.0
_C.SAN.SOLVER.LR_POLICY = 'cosine'
_C.SAN.SOLVER.COSINE_END_LR = 0.0
_C.SAN.SOLVER.COSINE_AFTER_WARMUP = False
_C.SAN.SOLVER.WARMUP_EPOCHS = 0
_C.SAN.SOLVER.WARMUP_START_LR = 0.001

_C.REVIN = CN()
_C.REVIN.EPS = 1e-5
_C.REVIN.AFFINE = True
_C.REVIN.RESULT_DIR = 'results/revin/'
_C.REVIN.TRAIN = CN()
_C.REVIN.TRAIN.CHECKPOINT_DIR = 'results/revin/'

_C.DISHTS = CN()
_C.DISHTS.INIT = 'standard'  # standard, avg, uniform
_C.DISHTS.RESULT_DIR = 'results/dishts/'
_C.DISHTS.TRAIN = CN()
_C.DISHTS.TRAIN.CHECKPOINT_DIR = 'results/dishts/'

_C.SOLVER = CN()
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.MAX_EPOCH = 30
_C.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = True
_C.SOLVER.DAMPENING = 0.0
_C.SOLVER.LR_POLICY = 'cosine'
_C.SOLVER.COSINE_END_LR = 0.0
_C.SOLVER.COSINE_AFTER_WARMUP = False
_C.SOLVER.WARMUP_EPOCHS = 0
_C.SOLVER.WARMUP_START_LR = 0.001


def get_cfg_defaults():

    return _C.clone()


def get_norm_module_cfg(cfg):
    return getattr(cfg, cfg.NORM_MODULE.NAME.upper())


def get_norm_method(cfg):
    assert cfg.NORM_MODULE.NAME in ('RevIN', 'SAN', 'DishTS')
    norm_method = cfg.NORM_MODULE.NAME if cfg.NORM_MODULE.ENABLE else 'NST'
    return norm_method
