import os
import sys
import time
import datetime
import argparse
import logging
import os.path as osp
import numpy as np
import multiprocessing
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import distributed as dist
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter


from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from tools.utils import save_checkpoint, set_seed, get_logger
from train import train_cal, train_cal_with_memory
from test import test, test_prcc, extract_features_with_clothes, visualize_tsne_with_clothes
from yacs.config import CfgNode as CN

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# CUDA와 멀티프로세싱 호환을 위해 'spawn' 방식 사용
multiprocessing.set_start_method('spawn', force=True)

VID_DATASET = ['ccvid']


def parse_option():
    parser = argparse.ArgumentParser('LTCC ReID Training')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='4', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    # Diffusion Augmentation
    parser.add_argument("--diffusion-aug", action="store_true", help="Enable diffusion augmentation")
    parser.add_argument("--aug-prob", type=float, default=0.3, help="Probability for diffusion augmentation")
    parser.add_argument("--sd-model", type=str, default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model name")
    parser.add_argument("--controlnet", type=str, default="lllyasviel/control_v11p_sd15_openpose", help="ControlNet model name")

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

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


def main():
    config = parse_option()

    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    
    # 분산 학습 초기화
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    
    # wandb 대신 tensorboard 사용
    if local_rank == 0:
        # tensorboard writer 초기화
        tb_log_dir = osp.join(config.OUTPUT, 'tensorboard')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
    else:
        writer = None
    
    # Set random seed
    set_seed(config.SEED + local_rank)
    # get logger
    if not config.EVAL_MODE:
        output_file = osp.join(config.OUTPUT, 'log_train_.log')
    else:
        output_file = osp.join(config.OUTPUT, 'log_test.log')
    logger = get_logger(output_file, local_rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")

    # config 객체 내용 출력
    print("Config:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    # Diffusion 관련 설정 출력
    print(f"Diffusion Augmentation Enabled: {getattr(config, 'DIFFUSION_AUG.ENABLED', False)}")
    print(f"Diffusion Augmentation Probability: {getattr(config, 'DIFFUSION_AUG.PROB', None)}")
    print(f"Stable Diffusion Model: {getattr(config, 'DIFFUSION_AUG.SD_MODEL', None)}")
    print(f"ControlNet Model: {getattr(config, 'DIFFUSION_AUG.CONTROLNET', None)}")

    # Build dataloader
    result = build_dataloader(config)
    

    # 데이터셋 종류에 관계없이 5개의 변수로 언패킹
    trainloader, queryloader, galleryloader, dataset, train_sampler = result

    # Define a matrix pid2clothes with shape (num_pids, num_clothes). 
    # pid2clothes[i, j] = 1 when j-th clothes belongs to i-th identity. Otherwise, pid2clothes[i, j] = 0.
    pid2clothes = torch.from_numpy(dataset.pid2clothes) # CPU에 생성됨

    # Move pid2clothes to the GPU
    # local_rank는 이전에 dist.get_rank()를 통해 설정되었습니다.
    pid2clothes = pid2clothes.cuda(local_rank, non_blocking=True) # GPU로 이동

    # Build model
    model, classifier, clothes_classifier = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
    # Build identity classification loss, pairwise loss, clothes classificaiton loss, and adversarial loss.
    criterion_cla, criterion_pair, criterion_clothes, criterion_adv = build_losses(config, dataset.num_train_clothes)
    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        optimizer_cc = optim.Adam(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, 
                                  weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        optimizer_cc = optim.AdamW(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, 
                                  weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
        optimizer_cc = optim.SGD(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        if config.LOSS.CAL == 'calwithmemory':
            criterion_adv.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        else:
            clothes_classifier.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        start_epoch = checkpoint['epoch']

    local_rank = dist.get_rank()
    model = model.cuda(local_rank)
    classifier = classifier.cuda(local_rank)
    if config.LOSS.CAL == 'calwithmemory':
        criterion_adv = criterion_adv.cuda(local_rank)
    else:
        clothes_classifier = clothes_classifier.cuda(local_rank)
    torch.cuda.set_device(local_rank)

    if config.TRAIN.AMP:
        pass

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank)
    if config.LOSS.CAL != 'calwithmemory':
        clothes_classifier = nn.parallel.DistributedDataParallel(clothes_classifier, device_ids=[local_rank], output_device=local_rank)

    if config.EVAL_MODE:
        logger.info("Evaluate only")
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                test(config, model, queryloader, galleryloader, dataset)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    best_model_state = None  # 최고 성능 모델 상태 저장용
    
    logger.info("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        train_sampler.set_epoch(epoch)
        start_train_time = time.time()
        if config.LOSS.CAL == 'calwithmemory':
            train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
                criterion_adv, optimizer, trainloader, pid2clothes)
        else:
            train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
                criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes)
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL:
            # writer 전달
            rank1 = test(config, model, queryloader, galleryloader, dataset, epoch, local_rank, writer)
    
    # tensorboard 종료
    if local_rank == 0 and writer is not None:
        writer.close()

if __name__ == '__main__':
    main()