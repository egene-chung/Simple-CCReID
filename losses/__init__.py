from torch import nn
from losses.dynamic_loss import DynamicLossWrapper
from losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from losses.triplet_loss import TripletLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.arcface_loss import ArcFaceLoss
from losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from losses.circle_loss import CircleLoss, PairwiseCircleLoss
from losses.clothes_based_adversarial_loss import ClothesBasedAdversarialLoss, ClothesBasedAdversarialLossWithMemoryBank


def build_losses(config, num_train_clothes):
    # Build identity classification loss
    if config.LOSS.CLA_LOSS == 'ce' or config.LOSS.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyWithLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M) 
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterion_pair = TripletLoss(margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterion_pair = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterion_pair = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterion_pair = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))

    # Build clothes classification loss
    if config.LOSS.CLOTHES_CLA_LOSS == 'crossentropy':
        criterion_clothes = nn.CrossEntropyLoss()
    elif config.LOSS.CLOTHES_CLA_LOSS == 'cosface':
        criterion_clothes = CosFaceLoss(scale=config.LOSS.CLA_S, margin=0)
    else:
        raise KeyError("Invalid clothes classification loss: '{}'".format(config.LOSS.CLOTHES_CLA_LOSS))

    # Build clothes-based adversarial loss
    if config.LOSS.CAL == 'cal':
        criterion_cal = ClothesBasedAdversarialLoss(
            scale=config.LOSS.SCALE,
            epsilon=config.LOSS.EPSILON,
            use_dynamic_epsilon=config.LOSS.DYNAMIC_EPSILON,
            alpha_scale=config.LOSS.ALPHA_SCALE,
            use_dual_temperature=config.LOSS.DUAL_TEMPERATURE,
            temp_min=config.LOSS.TEMP_MIN,
            temp_max=config.LOSS.TEMP_MAX,
            use_entropy_clip=config.LOSS.ENTROPY_CLIP,
            epsilon_min=config.LOSS.EPSILON_MIN,
            epsilon_max=config.LOSS.EPSILON_MAX
        )
    elif config.LOSS.CAL == 'cal_with_memory' or config.LOSS.CAL == 'calwithmemory':
        criterion_cal = ClothesBasedAdversarialLossWithMemoryBank(
            num_clothes=num_train_clothes,
            feat_dim=config.MODEL.FEATURE_DIM,
            momentum=config.LOSS.MOMENTUM,
            scale=config.LOSS.SCALE,
            epsilon=config.LOSS.EPSILON,
            use_dynamic_epsilon=config.LOSS.DYNAMIC_EPSILON,
            alpha_scale=config.LOSS.ALPHA_SCALE,
            use_dual_temperature=config.LOSS.DUAL_TEMPERATURE,
            temp_min=config.LOSS.TEMP_MIN,
            temp_max=config.LOSS.TEMP_MAX,
            use_entropy_clip=config.LOSS.ENTROPY_CLIP,
            epsilon_min=config.LOSS.EPSILON_MIN,
            epsilon_max=config.LOSS.EPSILON_MAX
        )
    else:
        criterion_cal = None

    return criterion_cla, criterion_pair, criterion_clothes, criterion_cal
