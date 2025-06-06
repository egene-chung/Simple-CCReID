import torch
import torch.nn.functional as F
from torch import nn
from losses.gather import GatherLayer
import math


class ClothesBasedAdversarialLoss(nn.Module):
    """ Clothes-based Adversarial Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
        use_dynamic_epsilon (bool): whether to use dynamic epsilon.
        alpha_scale (float): scaling factor for entropy difference.
        use_dual_temperature (bool): whether to use dynamic temperature.
        temp_min (float): minimum temperature for softmax.
        temp_max (float): maximum temperature for softmax.
        use_entropy_clip (bool): whether to clip entropy ratio.
        epsilon_min (float): minimum epsilon value after clipping.
        epsilon_max (float): maximum epsilon value after clipping.
    """
    def __init__(self, scale=16, epsilon=0.1, use_dynamic_epsilon=False, alpha_scale=10.0,
                 use_dual_temperature=False, temp_min=0.5, temp_max=2.0,
                 use_entropy_clip=False, epsilon_min=0.01, epsilon_max=0.3):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon
        self.use_dynamic_epsilon = use_dynamic_epsilon
        self.alpha_scale = alpha_scale
        
        # Dual Temperature 파라미터
        self.use_dual_temperature = use_dual_temperature
        self.temp_min = temp_min
        self.temp_max = temp_max
        
        # Entropy-Ratio Clipping 파라미터
        self.use_entropy_clip = use_entropy_clip
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        
    def compute_entropy(self, logits):
        """Compute entropy of softmax distribution"""
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy
        
    def compute_dynamic_epsilon(self, clothes_logits, id_logits, epoch=None, max_epoch=None):
        """
        Compute dynamic epsilon based on entropy difference
        
        Args:
            clothes_logits: logits for clothes classification
            id_logits: logits for ID classification
            epoch: current epoch
            max_epoch: total number of epochs
        """
        # Compute entropy for clothes and ID classification
        clothes_entropy = self.compute_entropy(clothes_logits)
        id_entropy = self.compute_entropy(id_logits)
        
        # Cosine scheduling for alpha
        if epoch is not None and max_epoch is not None:
            progress = epoch / max_epoch
            alpha_t = self.alpha_scale * (0.5 * (1 + math.cos(math.pi * progress)))
        else:
            alpha_t = self.alpha_scale
        
        # ε_i = sigmoid(α·(H_c(i) – H_id(i)))
        entropy_diff = clothes_entropy - id_entropy
        dynamic_epsilon = torch.sigmoid(alpha_t * entropy_diff)
        
        # Entropy-Ratio Clipping 적용
        if self.use_entropy_clip:
            dynamic_epsilon = torch.clamp(dynamic_epsilon, self.epsilon_min, self.epsilon_max)
            
        return dynamic_epsilon
        
    def compute_dynamic_temperature(self, clothes_logits, id_logits):
        """
        Compute dynamic temperature based on entropy difference
        
        Args:
            clothes_logits: logits for clothes classification
            id_logits: logits for ID classification
        """
        # Compute entropy for clothes and ID classification
        clothes_entropy = self.compute_entropy(clothes_logits)
        id_entropy = self.compute_entropy(id_logits)
        
        # 엔트로피 차이 기반 온도 계산
        # easy-positive (옷 특징이 약함) => 높은 온도 (gradient 완화)
        # hard-positive (옷 특징이 강함) => 낮은 온도 (분포 선명하게)
        entropy_ratio = clothes_entropy / (id_entropy + 1e-8)
        
        # 온도를 [temp_min, temp_max] 범위로 매핑
        temp = self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(1.0 - entropy_ratio)
        
        return temp

    def forward(self, inputs, targets, positive_mask, clothes_logits=None, id_logits=None, epoch=None, max_epoch=None):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with 
                the same identity as the anchor sample are defined as positive clothes classes and their mask 
                values are 1. The clothes classes with different identities from the anchor sample are defined 
                as negative clothes classes and their mask values in positive_mask are 0.
            clothes_logits: logits for clothes classification (optional, for dynamic epsilon)
            id_logits: logits for ID classification (optional, for dynamic epsilon)
            epoch: current epoch (optional, for cosine scheduling)
            max_epoch: total number of epochs (optional, for cosine scheduling)
        """
        # Dual Temperature 적용
        if self.use_dual_temperature and clothes_logits is not None and id_logits is not None:
            temperature = self.compute_dynamic_temperature(clothes_logits, id_logits).unsqueeze(1)
            # 온도로 inputs 스케일링 (높은 온도 = 더 부드러운 분포)
            scaled_inputs = self.scale * inputs / temperature
        else:
            scaled_inputs = self.scale * inputs
            
        negtive_mask = 1 - positive_mask
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        exp_logits = torch.exp(scaled_inputs)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = scaled_inputs - log_sum_exp_pos_and_all_neg

        # Compute dynamic epsilon if enabled
        if self.use_dynamic_epsilon and clothes_logits is not None and id_logits is not None:
            epsilon = self.compute_dynamic_epsilon(clothes_logits, id_logits, epoch, max_epoch).unsqueeze(1)
        else:
            epsilon = self.epsilon
            
        # Apply epsilon to create weighted mask
        mask = (1 - epsilon) * identity_mask + epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_prob).sum(1).mean()

        return loss


class ClothesBasedAdversarialLossWithMemoryBank(nn.Module):
    """ Clothes-based Adversarial Loss between mini batch and the samples in memory.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        num_clothes (int): the number of clothes classes.
        feat_dim (int): the dimensions of feature.
        momentum (float): momentum to update memory.
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
        use_dynamic_epsilon (bool): whether to use dynamic epsilon.
        alpha_scale (float): scaling factor for entropy difference.
        use_dual_temperature (bool): whether to use dynamic temperature.
        temp_min (float): minimum temperature for softmax.
        temp_max (float): maximum temperature for softmax.
        use_entropy_clip (bool): whether to clip entropy ratio.
        epsilon_min (float): minimum epsilon value after clipping.
        epsilon_max (float): maximum epsilon value after clipping.
    """
    def __init__(self, num_clothes, feat_dim, momentum=0., scale=16, epsilon=0.1, 
                 use_dynamic_epsilon=False, alpha_scale=10.0,
                 use_dual_temperature=False, temp_min=0.5, temp_max=2.0,
                 use_entropy_clip=False, epsilon_min=0.01, epsilon_max=0.3):
        super().__init__()
        self.num_clothes = num_clothes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = scale
        self.use_dynamic_epsilon = use_dynamic_epsilon
        self.alpha_scale = alpha_scale
        
        # Dual Temperature 파라미터
        self.use_dual_temperature = use_dual_temperature
        self.temp_min = temp_min
        self.temp_max = temp_max
        
        # Entropy-Ratio Clipping 파라미터
        self.use_entropy_clip = use_entropy_clip
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max

        self.register_buffer('feature_memory', torch.zeros((num_clothes, feat_dim)))
        self.register_buffer('label_memory', torch.zeros(num_clothes, dtype=torch.int64) - 1)
        self.has_been_filled = False
        
    def compute_entropy(self, logits):
        """Compute entropy of softmax distribution"""
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy
        
    def compute_dynamic_epsilon(self, clothes_logits, id_logits, epoch=None, max_epoch=None):
        """
        Compute dynamic epsilon based on entropy difference
        
        Args:
            clothes_logits: logits for clothes classification
            id_logits: logits for ID classification
            epoch: current epoch
            max_epoch: total number of epochs
        """
        # Compute entropy for clothes and ID classification
        clothes_entropy = self.compute_entropy(clothes_logits)
        id_entropy = self.compute_entropy(id_logits)
        
        # Cosine scheduling for alpha
        if epoch is not None and max_epoch is not None:
            progress = epoch / max_epoch
            alpha_t = self.alpha_scale * (0.5 * (1 + math.cos(math.pi * progress)))
        else:
            alpha_t = self.alpha_scale
        
        # ε_i = sigmoid(α·(H_c(i) – H_id(i)))
        entropy_diff = clothes_entropy - id_entropy
        dynamic_epsilon = torch.sigmoid(alpha_t * entropy_diff)
        
        # Entropy-Ratio Clipping 적용
        if self.use_entropy_clip:
            dynamic_epsilon = torch.clamp(dynamic_epsilon, self.epsilon_min, self.epsilon_max)
            
        return dynamic_epsilon
        
    def compute_dynamic_temperature(self, clothes_logits, id_logits):
        """
        Compute dynamic temperature based on entropy difference
        
        Args:
            clothes_logits: logits for clothes classification
            id_logits: logits for ID classification
        """
        # Compute entropy for clothes and ID classification
        clothes_entropy = self.compute_entropy(clothes_logits)
        id_entropy = self.compute_entropy(id_logits)
        
        # 엔트로피 차이 기반 온도 계산
        entropy_ratio = clothes_entropy / (id_entropy + 1e-8)
        
        # 온도를 [temp_min, temp_max] 범위로 매핑
        temp = self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(1.0 - entropy_ratio)
        
        return temp

    def forward(self, inputs, targets, positive_mask, clothes_logits=None, id_logits=None, epoch=None, max_epoch=None):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes).
            clothes_logits: logits for clothes classification (optional, for dynamic epsilon)
            id_logits: logits for ID classification (optional, for dynamic epsilon)
            epoch: current epoch (optional, for cosine scheduling)
            max_epoch: total number of epochs (optional, for cosine scheduling)
        """
        # gather all samples from different GPUs to update memory.
        gathered_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gathered_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        self._update_memory(gathered_inputs.detach(), gathered_targets)

        inputs_norm = F.normalize(inputs, p=2, dim=1)
        memory_norm = F.normalize(self.feature_memory.detach(), p=2, dim=1)
        
        # Dual Temperature 적용
        if self.use_dual_temperature and clothes_logits is not None and id_logits is not None:
            temperature = self.compute_dynamic_temperature(clothes_logits, id_logits).unsqueeze(1)
            # 온도로 similarity 스케일링
            similarities = torch.matmul(inputs_norm, memory_norm.t()) * self.scale / temperature
        else:
            similarities = torch.matmul(inputs_norm, memory_norm.t()) * self.scale

        negtive_mask = 1 - positive_mask
        mask_identity = torch.zeros(positive_mask.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        if not self.has_been_filled:
            invalid_index = self.label_memory == -1
            positive_mask[:, invalid_index] = 0
            negtive_mask[:, invalid_index] = 0
            if sum(invalid_index.type(torch.int)) == 0:
                self.has_been_filled = True
                print('Memory bank is full')

        # compute log_prob
        exp_logits = torch.exp(similarities)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = similarities - log_sum_exp_pos_and_all_neg

        # Compute dynamic epsilon if enabled
        if self.use_dynamic_epsilon and clothes_logits is not None and id_logits is not None:
            epsilon = self.compute_dynamic_epsilon(clothes_logits, id_logits, epoch, max_epoch).unsqueeze(1)
        else:
            epsilon = self.epsilon

        # compute mean of log-likelihood over positive with dynamic epsilon
        mask = (1 - epsilon) * mask_identity + epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_prob).sum(1).mean()
        
        return loss

    def _update_memory(self, features, labels):
        # 원래 _update_memory 함수 그대로 유지
        label_to_feat = {}
        for x, y in zip(features, labels):
            if y not in label_to_feat:
                label_to_feat[y] = [x.unsqueeze(0)]
            else:
                label_to_feat[y].append(x.unsqueeze(0))
        if not self.has_been_filled:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0)
                self.feature_memory[y] = feat
                self.label_memory[y] = y
        else:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0)
                self.feature_memory[y] = self.momentum * self.feature_memory[y] + (1. - self.momentum) * feat