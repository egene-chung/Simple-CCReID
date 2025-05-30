import time
import datetime
import logging
import torch
from tools.utils import AverageMeter
from losses import DynamicLossWrapper


def train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        pred_clothes = clothes_classifier(features.detach())
        _, preds = torch.max(outputs.data, 1)

        # Update the clothes discriminator
        clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            clothes_loss.backward()
            optimizer_cc.step()

        # Update the backbone
        new_pred_clothes = clothes_classifier(features)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'PairLoss:{pair_loss.avg:.4f} '
                  'CloLoss:{clo_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   acc=corrects, clo_acc=clothes_corrects))


def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
                         criterion_adv, optimizer, trainloader, pid2clothes, max_epoch=None):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    # Dynamic Loss Wrapper 초기화 (이 부분이 빠져있었음)
    if config.TRAIN.DYNAMIC_LOSS:
        # 모델이 DDP로 래핑된 경우 module에 접근
        actual_model = model.module if hasattr(model, 'module') else model
        
        # loss_wrapper가 없는 경우에만 초기화
        if not hasattr(actual_model, 'loss_wrapper'):
            print(f"Dynamic Loss Wrapper 초기화 중...")
            actual_model.loss_wrapper = DynamicLossWrapper(
                num_tasks=4,
                twa_temp=config.TRAIN.DYN_LOSS.TWA_TEMP
            ).cuda()
            # loss wrapper의 파라미터도 optimizer에 추가
            optimizer.add_param_group({'params': actual_model.loss_wrapper.parameters()})
            print(f"Dynamic Loss Wrapper 초기화 완료!")
    
    end = time.time()
    for idx, (imgs, pids, _, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)  # 모델에서 3개 값 반환
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)

        # adv_loss는 조건부로 계산
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            adv_loss = criterion_adv(features, clothes_ids, pos_mask)
        else:
            adv_loss = torch.tensor(0.0, device=features.device)  # 0으로 설정

        # Dynamic Loss는 워밍업 기간 이후 항상 적용
        if config.TRAIN.DYNAMIC_LOSS and epoch >= config.TRAIN.DYN_LOSS.WARMUP_EPOCHS:
            # 분산 모델 처리
            actual_model = model.module if hasattr(model, 'module') else model
            
            # 손실 사전 구성
            loss_dict = {
                'id': cla_loss,
                'pair': pair_loss,
                'var': (features - features.mean(0)).pow(2).sum(1).mean()
            }
            
            # adv_loss가 활성화된 경우에만 포함
            if epoch >= config.TRAIN.START_EPOCH_ADV:
                loss_dict['cal'] = adv_loss
                
            # Dynamic Loss Wrapper 호출
            loss = actual_model.loss_wrapper(epoch, loss_dict)
        else:
            # 일반적인 손실 합산
            if epoch >= config.TRAIN.START_EPOCH_ADV:
                loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss
            else:
                loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV: 
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                epoch+1, batch_time=batch_time, data_time=data_time, 
                cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                adv_loss=batch_adv_loss, acc=corrects))