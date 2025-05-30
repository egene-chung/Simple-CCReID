import os
import sys
import time
import datetime
import argparse
import logging
import glob
import re
import numpy as np
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.utils.data.sampler import Sampler
import torchvision.models as models
from PIL import Image
import random


# ---------------------- 로깅 유틸리티 ----------------------
def get_logger(output=None, level='INFO'):
    """
    로깅을 위한 로거 인스턴스를 반환합니다.
    args:
        output (str): 로그 파일 경로. None이면 콘솔에만 출력.
        level (str): 로그 레벨. INFO, WARNING, ERROR, DEBUG 중 하나.
    """
    logger = logging.getLogger('reid')
    level_dict = {
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'DEBUG': logging.DEBUG
    }
    level = level_dict.get(level, logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(level)
    
    # 콘솔 핸들러
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # 파일 핸들러 (지정된 경우)
    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        file_handler = logging.FileHandler(output)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed):
    """
    재현성을 위한 모든 랜덤 시드를 설정합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, fpath):
    """
    모델 체크포인트를 저장합니다.
    args:
        state (dict): 저장할 상태
        is_best (bool): 현재 모델이 최고성능인지 여부
        fpath (str): 저장 경로
    """
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    torch.save(state, fpath)
    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), 'best_model.pth.tar')
        torch.save(state, best_fpath)


# ---------------------- 데이터셋 및 데이터 로더 ----------------------
class LTCC(object):
    """
    LTCC 데이터셋을 로드합니다.
    
    Reference:
        Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.
    
    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'LTCC_ReID'
    
    def __init__(self, root='data', verbose=True):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        
        self._check_before_run()
        
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
        
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes
        
        if verbose:
            print("=> LTCC loaded")
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # clothes")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
            print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
            print("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
            print("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
            print("  ----------------------------------------")
        
        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
    
    def _check_before_run(self):
        """모든 파일이 사용 가능한지 확인합니다."""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    
    def _process_dir_train(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')
        
        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        
        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes = pattern2.search(img_path).group(1)
            camid -= 1  # index starts from 0
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            dataset.append((img_path, pid, camid, clothes_id))
            pid2clothes[pid, clothes_id] = 1
        
        num_imgs = len(dataset)
        
        return dataset, num_pids, num_imgs, num_clothes, pid2clothes
    
    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.png'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.png'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')
        
        pid_container = set()
        clothes_container = set()
        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        
        query_dataset = []
        gallery_dataset = []
        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1  # index starts from 0
            clothes_id = clothes2label[clothes_id]
            query_dataset.append((img_path, pid, camid, clothes_id))
        
        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1  # index starts from 0
            clothes_id = clothes2label[clothes_id]
            gallery_dataset.append((img_path, pid, camid, clothes_id))
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)
        
        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid, clothes_id


class RandomIdentitySampler(Sampler):
    """
    배치 내에서 각 ID마다 고정된 수의 샘플을 샘플링합니다.
    
    사용 예시:
        RandomIdentitySampler(dataset, batch_size=32, num_instances=4)
        이는 배치의 각 사람 ID마다 4개의 인스턴스를 선택합니다.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            if pid not in self.index_dic:
                self.index_dic[pid] = []
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    
    def __iter__(self):
        list_container = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    list_container.append(batch_idxs)
                    batch_idxs = []

        random.shuffle(list_container)

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)
    
    def __len__(self):
        return self.length


def build_transforms(img_size=(256, 128), is_train=True):
    """
    이미지 변환 파이프라인을 빌드합니다.
    """
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if is_train:
        transform = T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(img_size),
            T.ToTensor(),
            normalize_transform,
            T.RandomErasing(p=0.5)
        ])
    else:
        transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            normalize_transform
        ])
    
    return transform


def build_dataloader(dataset, img_size=(256, 128), batch_size=32, num_instances=4, workers=4):
    """
    데이터 로더를 구축합니다.
    """
    # LTCC 데이터셋 로드
    ltcc_dataset = LTCC(root=dataset)
    
    # 이미지 변환 생성
    train_transform = build_transforms(img_size, is_train=True)
    test_transform = build_transforms(img_size, is_train=False)
    
    # 이미지 데이터셋 생성
    train_set = ImageDataset(ltcc_dataset.train, train_transform)
    query_set = ImageDataset(ltcc_dataset.query, test_transform)
    gallery_set = ImageDataset(ltcc_dataset.gallery, test_transform)
    
    # 샘플러 생성
    train_sampler = RandomIdentitySampler(ltcc_dataset.train, batch_size, num_instances)
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler,
        num_workers=workers, pin_memory=True
    )
    query_loader = DataLoader(
        query_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    gallery_loader = DataLoader(
        gallery_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    
    return train_loader, query_loader, gallery_loader, ltcc_dataset


# ---------------------- 모델 구성 ----------------------
class ResNetBackbone(nn.Module):
    """ResNet backbone 모델"""
    def __init__(self, model_name='resnet50', pretrained=True, last_stride=1):
        super(ResNetBackbone, self).__init__()
        
        if model_name == 'resnet18':
            self.base = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            self.base = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.base = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            self.base = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet152':
            self.base = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError('Unsupported resnet model: {}'.format(model_name))
        
        # ResNet의 마지막 stride 수정
        if last_stride != 1:
            self.base.layer4[0].downsample[0].stride = (last_stride, last_stride)
            self.base.layer4[0].conv2.stride = (last_stride, last_stride)
        
        # global average pooling을 위한 레이어
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # fc 레이어 제거
        del self.base.fc
        del self.base.avgpool
    
    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        
        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (b, 2048)
        
        return global_feat


class ClassifierHead(nn.Module):
    """분류 헤드 모델"""
    def __init__(self, feature_dim, num_classes):
        super(ClassifierHead, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
    
    def forward(self, x):
        return self.classifier(x)


def build_model(model_name='resnet50', pretrained=True, num_classes=0, num_clothes_classes=0, last_stride=1):
    """
    모델을 구축합니다.
    """
    # 백본 모델 생성
    model = ResNetBackbone(model_name, pretrained, last_stride)
    feature_dim = model.feature_dim
    
    # ID 분류기 생성
    classifier = ClassifierHead(feature_dim, num_classes)
    
    # 의류 분류기 생성
    clothes_classifier = ClassifierHead(feature_dim, num_clothes_classes)
    
    return model, classifier, clothes_classifier


# ---------------------- 손실 함수 ----------------------
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer."""
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 예측 로짓
            targets: 대상 레이블
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1).mean()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining."""
    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 특징 매트릭스, shape [B, C]
            targets: 대상 레이블, shape [B]
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_ap)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss


class CALoss(nn.Module):
    """Clothes-based Adversarial Loss"""
    def __init__(self, alpha=1, neg_clothes_weight=1):
        super(CALoss, self).__init__()
        self.alpha = alpha
        self.neg_clothes_weight = neg_clothes_weight
    
    def forward(self, inputs, clothes_labels, pid2clothes):
        """
        Args:
            inputs: 의류 분류 점수, shape [B, num_clothes]
            clothes_labels: 의류 레이블, shape [B]
            pid2clothes: 사람-의류 관계 매트릭스, shape [num_pids, num_clothes]
        """
        batch_size = inputs.size(0)
        clothes_onehot = torch.zeros(batch_size, pid2clothes.size(1)).to(inputs.device)
        clothes_onehot.scatter_(1, clothes_labels.view(-1, 1), 1)
        
        loss_adv = -torch.mean(clothes_onehot * F.log_softmax(inputs, dim=1))
        
        return self.alpha * loss_adv


def build_losses(num_classes, num_clothes_classes, margin=0.3, label_smooth=True):
    """손실 함수를 구축합니다."""
    if label_smooth:
        criterion_cla = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        criterion_cla = nn.CrossEntropyLoss()
    
    criterion_pair = TripletLoss(margin=margin)
    criterion_clothes = nn.CrossEntropyLoss()
    criterion_adv = CALoss(alpha=0.1)
    
    return criterion_cla, criterion_pair, criterion_clothes, criterion_adv


# ---------------------- 학습 및 테스트 함수 ----------------------
def train_epoch(model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
               criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, 
               pid2clothes, epoch, start_clothes_cls_epoch=10, start_adv_epoch=25, 
               log_period=50, device=None):
    """한 에폭 동안 학습을 수행합니다."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.train()
    classifier.train()
    clothes_classifier.train()
    
    losses = 0
    cla_losses = 0
    pair_losses = 0
    clothes_cls_losses = 0
    adv_losses = 0
    
    for batch_idx, (imgs, pids, _, clothes_ids) in enumerate(trainloader):
        imgs, pids, clothes_ids = imgs.to(device), pids.to(device), clothes_ids.to(device)
        
        # 특징 추출
        features = model(imgs)
        
        # ID 분류 손실
        outputs = classifier(features)
        loss_cla = criterion_cla(outputs, pids)
        
        # 트리플렛 손실
        loss_pair = criterion_pair(features, pids)
        
        # 전체 손실
        loss = loss_cla + loss_pair
        
        # 의류 분류 손실 (epoch >= start_clothes_cls_epoch인 경우)
        if epoch >= start_clothes_cls_epoch:
            clothes_outputs = clothes_classifier(features.detach())
            loss_clothes = criterion_clothes(clothes_outputs, clothes_ids)
            optimizer_cc.zero_grad()
            loss_clothes.backward()
            optimizer_cc.step()
            clothes_cls_losses += loss_clothes.item()
        
        # 적대적 손실 (epoch >= start_adv_epoch인 경우)
        if epoch >= start_adv_epoch:
            clothes_outputs = clothes_classifier(features)
            loss_adv = criterion_adv(clothes_outputs, clothes_ids, pid2clothes)
            loss += loss_adv
            adv_losses += loss_adv.item()
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
        cla_losses += loss_cla.item()
        pair_losses += loss_pair.item()
        
        # 로그 출력
        if (batch_idx + 1) % log_period == 0:
            if epoch >= start_adv_epoch:
                print(f"Epoch: [{epoch}][{batch_idx+1}/{len(trainloader)}], "
                      f"Loss: {losses/(batch_idx+1):.4f}, "
                      f"Cla: {cla_losses/(batch_idx+1):.4f}, "
                      f"Pair: {pair_losses/(batch_idx+1):.4f}, "
                      f"ClothCls: {clothes_cls_losses/(batch_idx+1):.4f}, "
                      f"Adv: {adv_losses/(batch_idx+1):.4f}")
            elif epoch >= start_clothes_cls_epoch:
                print(f"Epoch: [{epoch}][{batch_idx+1}/{len(trainloader)}], "
                      f"Loss: {losses/(batch_idx+1):.4f}, "
                      f"Cla: {cla_losses/(batch_idx+1):.4f}, "
                      f"Pair: {pair_losses/(batch_idx+1):.4f}, "
                      f"ClothCls: {clothes_cls_losses/(batch_idx+1):.4f}")
            else:
                print(f"Epoch: [{epoch}][{batch_idx+1}/{len(trainloader)}], "
                      f"Loss: {losses/(batch_idx+1):.4f}, "
                      f"Cla: {cla_losses/(batch_idx+1):.4f}, "
                      f"Pair: {pair_losses/(batch_idx+1):.4f}")
    
    return {
        'loss': losses / len(trainloader),
        'cla_loss': cla_losses / len(trainloader),
        'pair_loss': pair_losses / len(trainloader),
        'clothes_cls_loss': clothes_cls_losses / len(trainloader) if epoch >= start_clothes_cls_epoch else 0,
        'adv_loss': adv_losses / len(trainloader) if epoch >= start_adv_epoch else 0
    }


def extract_features(model, dataloader, device=None):
    """테스트셋에서 특징을 추출합니다."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    features = []
    pids = []
    camids = []
    clothes_ids = []
    
    with torch.no_grad():
        for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
            imgs = imgs.to(device)
            batch_features = model(imgs)
            
            features.append(batch_features.cpu())
            pids.extend(batch_pids.numpy())
            camids.extend(batch_camids.numpy())
            clothes_ids.extend(batch_clothes_ids.numpy())
    
    features = torch.cat(features, 0)
    
    return features.numpy(), np.array(pids), np.array(camids), np.array(clothes_ids)


def compute_distance_matrix(query_features, gallery_features, metric='euclidean'):
    """
    쿼리와 갤러리 샘플 간의 거리 행렬을 계산합니다.
    args:
        query_features (numpy.ndarray): shape [m, d]
        gallery_features (numpy.ndarray): shape [n, d]
        metric (str): 'euclidean' 또는 'cosine'
    returns:
        거리 행렬 (numpy.ndarray): shape [m, n]
    """
    m, n = query_features.shape[0], gallery_features.shape[0]
    
    if metric == 'euclidean':
        # Euclidean distance
        query_features_norm = np.sum(np.square(query_features), axis=1, keepdims=True)
        gallery_features_norm = np.sum(np.square(gallery_features), axis=1, keepdims=True)
        
        dist = query_features_norm + gallery_features_norm.T - 2 * np.matmul(query_features, gallery_features.T)
        dist = np.sqrt(dist)
    
    elif metric == 'cosine':
        # Cosine distance
        query_features_norm = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
        gallery_features_norm = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
        
        dist = -np.matmul(query_features_norm, gallery_features_norm.T)
    
    return dist


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    """
    검색 결과를 평가합니다.
    
    Args:
        distmat (numpy.ndarray): 쿼리-갤러리 거리 행렬, shape [num_query, num_gallery]
        q_pids (numpy.ndarray): 쿼리 사람 IDs, shape [num_query]
        g_pids (numpy.ndarray): 갤러리 사람 IDs, shape [num_gallery]
        q_camids (numpy.ndarray): 쿼리 카메라 IDs, shape [num_query]
        g_camids (numpy.ndarray): 갤러리 카메라 IDs, shape [num_gallery]
    
    Returns:
        Rank-1, 5, 10, mAP
    """
    num_q, num_g = distmat.shape
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # compute cmc curve
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this query has no valid matches, skip it
            continue
        
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:10])
        
        # compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1
    
    if num_valid_q == 0:
        return 0, 0, 0, 0
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc[0], all_cmc[4], all_cmc[9], mAP


def test(model, queryloader, galleryloader, device=None):
    """모델을 테스트하고 성능을 반환합니다."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 특징 추출
    print("Extracting query features...")
    qf, q_pids, q_camids, q_clothes_ids = extract_features(model, queryloader, device)
    print("Extracting gallery features...")
    gf, g_pids, g_camids, g_clothes_ids = extract_features(model, galleryloader, device)
    
    # 거리 행렬 계산
    print("Computing distance matrix...")
    distmat = compute_distance_matrix(qf, gf, metric='euclidean')
    
    # 평가
    print("Evaluating...")
    # 수정된 부분: 4개의 반환값을 모두 받도록 변경
    rank1, rank5, rank10, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    print("Results ----------")
    print(f"mAP: {mAP:.2%}")
    print(f"CMC curve")
    print(f"Rank-1: {rank1:.2%}")
    print(f"Rank-5: {rank5:.2%}")
    print(f"Rank-10: {rank10:.2%}")
    print("------------------")
    
    return rank1


# ---------------------- 메인 함수 ----------------------
def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='LTCC ReID with ResNet and CAL')
    
    # 데이터셋 관련
    parser.add_argument('--root', type=str, default='data', help='dataset root path')
    parser.add_argument('--img-size', type=str, default='256,128', help='image size (height,width)')
    
    # 모델 관련
    parser.add_argument('--model', type=str, default='resnet50', help='model name (resnet18, resnet34, resnet50, etc.)')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--last-stride', type=int, default=1, help='last stride of the resnet')
    
    # 학습 관련
    parser.add_argument('--batch-size', type=int, default=32, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=128, help='test batch size')
    parser.add_argument('--num-instances', type=int, default=4, help='number of instances per identity in a batch')
    parser.add_argument('--max-epoch', type=int, default=60, help='maximum epochs to train')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--start-eval', type=int, default=0, help='start to evaluate after specific epoch')
    parser.add_argument('--eval-step', type=int, default=10, help='evaluation step')
    parser.add_argument('--start-clothes-cls', type=int, default=10, help='start epoch for clothes classification')
    parser.add_argument('--start-adv', type=int, default=25, help='start epoch for adversarial training')
    parser.add_argument('--lr', type=float, default=0.00035, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--margin', type=float, default=0.3, help='triplet loss margin')
    
    # 기타
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log-period', type=int, default=50, help='log period')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--output', type=str, default='output', help='output directory')
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='evaluation only')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
    
    args = parser.parse_args()
    
    # 랜덤 시드 설정
    set_seed(args.seed)
    
    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 로거 설정
    log_file = osp.join(args.output, 'log.txt')
    logger = get_logger(log_file)
    logger.info(args)
    
    # 이미지 크기 구문 분석
    img_h, img_w = map(int, args.img_size.split(','))
    img_size = (img_h, img_w)
    
    # 데이터로더 구축
    logger.info("Building dataloader...")
    train_loader, query_loader, gallery_loader, dataset = build_dataloader(
        args.root, img_size, args.batch_size, args.num_instances, args.workers
    )
    num_classes = dataset.num_train_pids
    num_clothes_classes = dataset.num_train_clothes
    pid2clothes = torch.from_numpy(dataset.pid2clothes).to(device)
    
    # 모델 구축
    logger.info("Building model...")
    model, classifier, clothes_classifier = build_model(
        args.model, args.pretrained, num_classes, num_clothes_classes, args.last_stride
    )
    
    model = model.to(device)
    classifier = classifier.to(device)
    clothes_classifier = clothes_classifier.to(device)
    
    # 손실 함수 구축
    logger.info("Building loss functions...")
    criterion_cla, criterion_pair, criterion_clothes, criterion_adv = build_losses(
        num_classes, num_clothes_classes, args.margin
    )
    
    # 옵티마이저 구축
    parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cc = optim.Adam(clothes_classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 스케줄러 구축
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    
    # 체크포인트 불러오기
    if args.resume:
        logger.info(f"Loading checkpoint from '{args.resume}'")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        clothes_classifier.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    # 평가 모드인 경우
    if args.eval_only:
        logger.info("Evaluation only")
        rank1 = test(model, query_loader, gallery_loader, device)
        return
    
    # 학습 시작
    logger.info("Starting training")
    best_rank1 = 0.0
    for epoch in range(args.start_epoch, args.max_epoch):
        # Train
        train_stats = train_epoch(
            model, classifier, clothes_classifier, criterion_cla, criterion_pair,
            criterion_clothes, criterion_adv, optimizer, optimizer_cc, train_loader,
            pid2clothes, epoch, args.start_clothes_cls, args.start_adv,
            args.log_period, device
        )
        
        # 로그 기록
        logger.info(f"Epoch {epoch}: {train_stats}")
        
        # 학습률 스케줄러 업데이트
        scheduler.step()
        
        # 평가
        if ((epoch+1) > args.start_eval) and (args.eval_step > 0) and ((epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch):
            logger.info("Evaluating...")
            rank1 = test(model, query_loader, gallery_loader, device)
            
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1
                logger.info(f"New best rank-1: {best_rank1:.2%} at epoch {best_epoch}")
            
            # 체크포인트 저장
            save_checkpoint(
                {
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'clothes_classifier_state_dict': clothes_classifier.state_dict(),
                    'rank1': rank1,
                    'epoch': epoch,
                },
                is_best,
                osp.join(args.output, f'checkpoint_ep{epoch+1}.pth.tar')
            )
    
    logger.info(f"Best rank-1: {best_rank1:.2%}, achieved at epoch {best_epoch}")


if __name__ == '__main__':
    import copy  # RandomIdentitySampler에서 사용됨
    main()