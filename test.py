import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


VID_DATASET = ['ccvid']


def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs


@torch.no_grad()
def extract_img_feature(model, dataloader):
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs)
        batch_features_flip = model(flip_imgs)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids


@torch.no_grad()
def extract_vid_feature(model, dataloader, vid2clip_index, data_length):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (vids, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        if (batch_idx + 1) % 200==0:
            logger.info("{}/{}".format(batch_idx+1, len(dataloader)))
        vids = vids.cuda()
        batch_features = model(vids)
        clip_features.append(batch_features.cpu())
        clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video
    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
    clip_features = clip_features.cuda()
    pids = torch.zeros(len(vid2clip_index))
    camids = torch.zeros(len(vid2clip_index))
    clothes_ids = torch.zeros(len(vid2clip_index))
    for i, idx in enumerate(vid2clip_index):
        features[i] = clip_features[idx[0] : idx[1], :].mean(0)
        features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids


def test(config, model, queryloader, galleryloader, dataset, current_epoch, local_rank, writer=None):
    logger = logging.getLogger('reid.test')
    model.eval()
    
    # 시간 측정 시작
    start_time = time.time()
    
    # Extract features 
    if config.DATA.DATASET in VID_DATASET:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, queryloader, 
                                                                  dataset.query_vid2clip_index,
                                                                  len(dataset.recombined_query))
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature(model, galleryloader, 
                                                                  dataset.gallery_vid2clip_index,
                                                                  len(dataset.recombined_gallery))
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(model, queryloader)
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
        # Gather samples from different GPUs
        torch.cuda.empty_cache()
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    torch.cuda.empty_cache()
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    
    # 거리 계산 후 시간 계산
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    logger.info("Computing CMC and mAP")
    # 일반적인 평가 결과 기록
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    

    if config.DATA.DATASET in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']: 
        return cmc[0]

    # Same clothes 설정 결과 기록
    cmc_sc, mAP_sc = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')

    # Clothes changing 결과 기록
    cmc_cc, mAP_cc = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')

    # 평가 결과 기록 - epoch 정보 포함
    if dist.get_rank() == 0:  # main process에서만 로깅
        metrics = {
            'epoch': current_epoch,
            'clothes_changing/mAP': mAP_cc * 100,
            'clothes_changing/top1': cmc_cc[0] * 100,
            'clothes_changing/top5': cmc_cc[4] * 100,
            'clothes_changing/top10': cmc_cc[9] * 100,
            'clothes_changing/top20': cmc_cc[19] * 100,
            "overall/mAP": mAP * 100,
            "overall/top1": cmc[0] * 100,
            "overall/top5": cmc[4] * 100,
            "overall/top10": cmc[9] * 100,
            "overall/top20": cmc[19] * 100,
            "same_clothes/mAP": mAP_sc * 100,
            "same_clothes/top1": cmc_sc[0] * 100,
            "same_clothes/top5": cmc_sc[4] * 100,
            "same_clothes/top10": cmc_sc[9] * 100,
            "same_clothes/top20": cmc_sc[19] * 100
        }
        # 결과 기록 - 메인 프로세스에서만
        if local_rank == 0:        
            logger.info(f"Epoch {current_epoch}:")
            logger.info(f"  clothes_changing/mAP: {mAP_cc*100:.2f}")
            logger.info(f"  clothes_changing/top1: {cmc_cc[0]*100:.2f}")
            logger.info(f"  clothes_changing/top5: {cmc_cc[4]*100:.2f}")
            logger.info(f"  clothes_changing/top10: {cmc_cc[9]*100:.2f}")
            logger.info(f"  clothes_changing/top20: {cmc_cc[19]*100:.2f}")
            logger.info(f"  overall/mAP: {mAP*100:.2f}")
            logger.info(f"  overall/top1: {cmc[0]*100:.2f}")
            logger.info(f"  overall/top5: {cmc[4]*100:.2f}")
            logger.info(f"  overall/top10: {cmc[9]*100:.2f}")
            logger.info(f"  overall/top20: {cmc[19]*100:.2f}")
            logger.info(f"  same_clothes/mAP: {mAP_sc*100:.2f}")
            logger.info(f"  same_clothes/top1: {cmc_sc[0]*100:.2f}")
            logger.info(f"  same_clothes/top5: {cmc_sc[4]*100:.2f}")
            logger.info(f"  same_clothes/top10: {cmc_sc[9]*100:.2f}")
            logger.info(f"  same_clothes/top20: {cmc_sc[19]*100:.2f}")

    # tensorboard에 로깅
    if local_rank == 0 and writer is not None:
        # 의류 변경 시나리오 메트릭
        writer.add_scalar('clothes_changing/mAP', mAP_cc * 100, current_epoch)
        writer.add_scalar('clothes_changing/top1', cmc_cc[0] * 100, current_epoch)
        writer.add_scalar('clothes_changing/top5', cmc_cc[4] * 100, current_epoch)
        writer.add_scalar('clothes_changing/top10', cmc_cc[9] * 100, current_epoch)
        writer.add_scalar('clothes_changing/top20', cmc_cc[19] * 100, current_epoch)
        
        # 전체 메트릭
        writer.add_scalar('overall/mAP', mAP * 100, current_epoch)
        writer.add_scalar('overall/top1', cmc[0] * 100, current_epoch)
        writer.add_scalar('overall/top5', cmc[4] * 100, current_epoch)
        writer.add_scalar('overall/top10', cmc[9] * 100, current_epoch)
        writer.add_scalar('overall/top20', cmc[19] * 100, current_epoch)
        
        # 같은 의류 시나리오 메트릭
        writer.add_scalar('same_clothes/mAP', mAP_sc * 100, current_epoch)
        writer.add_scalar('same_clothes/top1', cmc_sc[0] * 100, current_epoch)
        writer.add_scalar('same_clothes/top5', cmc_sc[4] * 100, current_epoch)
        writer.add_scalar('same_clothes/top10', cmc_sc[9] * 100, current_epoch)
        writer.add_scalar('same_clothes/top20', cmc_sc[19] * 100, current_epoch)
    
    # 로그에 출력
    logger.info(f"Epoch {current_epoch}:")
    logger.info(f"  clothes_changing/mAP: {mAP_cc*100:.2f}")
    logger.info(f"  clothes_changing/top1: {cmc_cc[0]*100:.2f}")
    logger.info(f"  overall/mAP: {mAP*100:.2f}")
    logger.info(f"  overall/top1: {cmc[0]*100:.2f}")
    logger.info(f"  same_clothes/mAP: {mAP_sc*100:.2f}")
    logger.info(f"  same_clothes/top1: {cmc_sc[0]*100:.2f}")
    
    return cmc[0]


def test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature(model, queryloader_same)
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature(model, queryloader_diff)
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))

    return cmc[0]


def visualize_tsne(features, labels, save_path, num_classes=10):
    """t-SNE 시각화 함수"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 처음 num_classes개의 클래스만 선택
    unique_labels = np.unique(labels)[:num_classes]
    mask = np.isin(labels, unique_labels)
    
    features_2d = features_2d[mask]
    labels = labels[mask]
    
    plt.figure(figsize=(10, 10))
    for label in unique_labels:
        idx = labels == label
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f'ID {label}')
    
    plt.title('t-SNE visualization of ReID features')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def extract_features(model, dataloader, num_samples=1000):
    """특징 벡터 추출 함수"""
    model.eval()
    features = []
    labels = []
    count = 0
    
    with torch.no_grad():
        for imgs, pids, _, _ in dataloader:
            if count >= num_samples:
                break
            imgs = imgs.cuda()
            feat = model(imgs)
            # CPU로 이동 후 NumPy 변환
            features.append(feat.cpu().numpy())  
            labels.append(pids.cpu().numpy())    # CPU로 이동 후 NumPy 변환
            count += imgs.size(0)
    
    return np.vstack(features), np.concatenate(labels)


def extract_features_with_clothes(model, dataloader):
    """특징 벡터와 ID, 옷 정보 추출 - 모든 샘플 포함"""
    model.eval()
    features = []
    person_ids = []
    clothes_ids = []
    
    with torch.no_grad():
        for imgs, pids, clothes, _ in dataloader:
            imgs = imgs.cuda()
            feat = model(imgs)
            features.append(feat.cpu().numpy())
            person_ids.append(pids.cpu().numpy())
            clothes_ids.append(clothes.cpu().numpy())
    
    return np.vstack(features), np.concatenate(person_ids), np.concatenate(clothes_ids)


def visualize_tsne_with_clothes(features, person_ids, clothes_ids, save_path, num_classes=10):
    """ID와 옷 정보를 포함한 t-SNE 시각화 - 선택된 ID의 모든 샘플 표시"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 가장 많은 샘플을 가진 ID들을 선택
    unique_pids = np.unique(person_ids)
    pid_counts = [(pid, sum(person_ids == pid)) for pid in unique_pids]
    pid_counts.sort(key=lambda x: x[1], reverse=True)
    selected_pids = [pid for pid, _ in pid_counts[:num_classes]]
    
    mask = np.isin(person_ids, selected_pids)
    features_2d = features_2d[mask]
    person_ids = person_ids[mask]
    clothes_ids = clothes_ids[mask]
    
    plt.figure(figsize=(15, 10))
    
    # 각 ID별로 다른 색상, 각 옷별로 다른 마커 사용
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '8']
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_pids)))
    
    for idx, pid in enumerate(selected_pids):
        pid_mask = person_ids == pid
        pid_clothes = np.unique(clothes_ids[pid_mask])
        total_samples = sum(pid_mask)
        
        for i, cid in enumerate(pid_clothes):
            mask = (person_ids == pid) & (clothes_ids == cid)
            samples = sum(mask)
            marker = markers[i % len(markers)]
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       color=colors[idx],
                       marker=marker,
                       label=f'ID {pid} - 옷 {cid} ({samples}개)')
    
    plt.title('t-SNE visualization (ID and Clothes)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              title='ID - 옷 (샘플 수)',
              ncol=1 + len(selected_pids) // 20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()