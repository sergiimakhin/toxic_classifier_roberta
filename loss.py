import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-10, max=10)
        
        # Use BCE with logits for numerical stability
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Convert logits to probabilities for focal weight calculation
        probs = torch.sigmoid(logits)
        
        # Calculate focal weight with numerical stability
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """Alternative simpler loss function"""
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)