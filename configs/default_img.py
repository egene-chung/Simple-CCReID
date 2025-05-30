import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root path for dataset directory
_C.DATA.ROOT = '/data1/egene/ccreid/'

# Dataset for evaluation
_C.DATA.DATASET = 'ltcc'
# Workers for dataloader
_C.DATA.NUM_WORKERS = 4
# Height of input image
_C.DATA.HEIGHT = 384
# Width of input image
_C.DATA.WIDTH = 192
# Batch size for training
_C.DATA.TRAIN_BATCH = 32
# Batch size for testing
_C.DATA.TEST_BATCH = 128
# The number of instances per identity for training sampler
_C.DATA.NUM_INSTANCES = 8
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Random crop prob
_C.AUG.RC_PROB = 0.5
# Random erase prob
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'resnet50'
# The stride for laery4 in resnet
_C.MODEL.RES4_STRIDE = 1
# feature dim
_C.MODEL.FEATURE_DIM = 4096
# Model path for resuming
_C.MODEL.RESUME = ''
# Global pooling after the backbone
_C.MODEL.POOLING = CN()
# Choose in ['avg', 'max', 'gem', 'maxavg']
_C.MODEL.POOLING.NAME = 'maxavg'
# Initialized power for GeM pooling
_C.MODEL.POOLING.P = 3
# -----------------------------------------------------------------------------
# Losses for training 
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# ID Loss
_C.LOSS.ID = 'ce'
# Clothes classification loss
_C.LOSS.CLA_LOSS = 'ce'
# Scale for classification loss
_C.LOSS.CLA_S = 16.
# Margin for classification loss
_C.LOSS.CLA_M = 0.
# Pairwise loss
_C.LOSS.PAIR_LOSS = 'triplet'
# The weight for pairwise loss
_C.LOSS.PAIR_LOSS_WEIGHT = 0.0
# Scale for pairwise loss
_C.LOSS.PAIR_S = 16.
# Margin for pairwise loss
_C.LOSS.PAIR_M = 0.3
# Clothes classification loss
_C.LOSS.CLOTHES_CLA_LOSS = 'crossentropy'
# Softmax epsilon
_C.LOSS.EPSILON = 0.1
# Clothes-based adversarial loss
_C.LOSS.CAL = 'cal'
# _C.LOSS.CAL = 'calwithmemory'
# Scale for clothes-based adversarial loss
_C.LOSS.SCALE = 16
# 동적 epsilon 사용 여부
_C.LOSS.DYNAMIC_EPSILON = True
# 엔트로피 차이 스케일링 파라미터
_C.LOSS.ALPHA_SCALE = 10.0
# Memory bank momentum
_C.LOSS.MOMENTUM = 0.1
# Dual Temperature 사용 여부
_C.LOSS.DUAL_TEMPERATURE = True
# 최소 온도값
_C.LOSS.TEMP_MIN = 0.5
# 최대 온도값
_C.LOSS.TEMP_MAX = 2.0

# Entropy-Ratio Clipping 사용 여부
_C.LOSS.ENTROPY_CLIP = True
# 최소 epsilon 값
_C.LOSS.EPSILON_MIN = 0.01
# 최대 epsilon 값
_C.LOSS.EPSILON_MAX = 0.3
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 60
# Start epoch for clothes classification
_C.TRAIN.START_EPOCH_CC = 25
# Start epoch for adversarial training
_C.TRAIN.START_EPOCH_ADV = 25
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 0.00035
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# Stepsize to decay learning rate
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [20, 40]
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Using amp for training
_C.TRAIN.AMP = False

# Dynamic Loss Wrapper 설정
_C.TRAIN.DYNAMIC_LOSS = True  # Dynamic Loss 사용 여부
_C.TRAIN.DYN_LOSS = CN()
_C.TRAIN.DYN_LOSS.TWA_TEMP = 2  # DWA temperature
_C.TRAIN.DYN_LOSS.ETA = 0.05    # 가중치 업데이트 속도
_C.TRAIN.DYN_LOSS.WARMUP_EPOCHS = 5  # 가중치 안정화 기간
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Perform evaluation after every N epochs (set to -1 to test after training)
_C.TEST.EVAL_STEP = 5
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 1
# Perform evaluation only
_C.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '0'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = 'logs/'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'res50-ce-cal'


def update_config(config, args):
    config.defrost()
    config.merge_from_file(args.cfg)

    # merge from specific arguments
    if args.root:
        config.DATA.ROOT = args.root
    if args.output:
        config.OUTPUT = args.output

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.eval:
        config.EVAL_MODE = True
    
    if args.tag:
        config.TAG = args.tag

    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.gpu:
        config.GPU = args.gpu
    if args.amp:
        # config.TRAIN.AMP = True
        config.TRAIN.AMP = False

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG)

    config.freeze()


def get_img_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    # Make config mutable
    config.defrost()

    # Diffusion Augmentation 설정 추가
    config.DIFFUSION_AUG = CN()
    config.DIFFUSION_AUG.ENABLED = args.diffusion_aug
    config.DIFFUSION_AUG.PROB = args.aug_prob
    config.DIFFUSION_AUG.SD_MODEL = args.sd_model
    config.DIFFUSION_AUG.CONTROLNET = args.controlnet

    # Freeze config again
    config.freeze()

    return config
