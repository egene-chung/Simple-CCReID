import torch
import torch.nn as nn
from collections import deque
import torch.distributed as dist

class DynamicLossWrapper(nn.Module):
    """손실 함수들을 동적 가중치로 결합하는 래퍼 클래스"""
    def __init__(self, num_tasks=4, twa_temp=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))  # [ID, CAL, Mean, Var]
        self.num_tasks = num_tasks
        self.twa_temp = twa_temp
        self.loss_history = [deque(maxlen=twa_temp) for _ in range(num_tasks)]
        
    def forward(self, epoch, loss_dict):
        losses = [
            loss_dict['id'],     # ID CE Loss
            loss_dict['cal'],    # Clothes Adversarial Loss
            loss_dict['mean'],   # Mean Alignment Loss  
            loss_dict['var']     # Variance Compactness Loss
        ]
        
        # DWA (Dynamic Weight Averaging)
        if epoch >= 2:
            w = []
            for i in range(self.num_tasks):
                prev2, prev1 = self.loss_history[i][0], self.loss_history[i][1]
                w.append(prev1 / (prev2 + 1e-8))
            w = torch.tensor(w, device=losses[0].device)
            w = self.num_tasks * torch.exp(w) / torch.exp(w).sum()
            
            with torch.no_grad():
                self.log_vars.add_((w.log() - self.log_vars) * 0.05)
        
        # Uncertainty Weighting
        precision = torch.exp(-self.log_vars)
        weighted = torch.stack(losses) * precision
        total_loss = weighted.sum() + self.log_vars.sum()
        
        # 히스토리 업데이트
        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss.detach())
        
        # 로깅
        lambdas = precision.detach().cpu().numpy()
        log_str = " | ".join([f"λ_{k}:{l:.3f}" for k, l in 
                            zip(['id', 'cal', 'mean', 'var'], lambdas)])
        print(f"[Epoch {epoch}] total:{total_loss.item():.4f} {log_str}")
        
        return total_loss