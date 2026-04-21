import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ListMLELoss(nn.Module):
    
    def __init__(self, temperature: float = 1.0):
        super(ListMLELoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:

        if torch.isnan(scores).any():
            return torch.tensor(0.0, device=scores.device)
        
        sorted_indices = torch.argsort(labels, descending=True)
        sorted_scores = scores[sorted_indices] / self.temperature
        
        loss = torch.tensor(0.0, device=scores.device)
        
        for i in range(len(sorted_scores)):
            remaining = sorted_scores[i:]
            loss += torch.logsumexp(remaining, dim=0) - sorted_scores[i]
        
        return loss / max(len(scores), 1)


class ListNetLoss(nn.Module):

    
    def __init__(self, temperature: float = 1.0):
        super(ListNetLoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:

        if torch.isnan(scores).any():
            return torch.tensor(0.0, device=scores.device)
        
        pred_probs = F.softmax(scores / self.temperature, dim=0)
        true_probs = F.softmax(labels.float() / self.temperature, dim=0)
        
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-8))
        
        return loss


class BCERankingLoss(nn.Module):

    
    def __init__(self):
        super(BCERankingLoss, self).__init__()
    
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:

        if torch.isnan(scores).any():
            return torch.tensor(0.0, device=scores.device)
        
        labels_min = labels.min()
        labels_max = labels.max()
        
        if labels_max > labels_min:
            labels_normalized = (labels - labels_min) / (labels_max - labels_min)
        else:
            labels_normalized = torch.ones_like(labels) * 0.5
        
        loss = F.binary_cross_entropy_with_logits(
            scores, labels_normalized.float()
        )
        
        return loss


class InfoNCELoss(nn.Module):

    
    def __init__(self, temperature: float = 0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:

        if torch.isnan(anchor).any():
            return torch.tensor(0.0, device=anchor.device)
        
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature
        
        # InfoNCE
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class CombinedLoss(nn.Module):

    
    def __init__(
        self,
        ranking_loss_type: str = 'bce',
        temperature: float = 1.0,
        lambda_completion: float = 0.01,
    ):
        super(CombinedLoss, self).__init__()
        
        if ranking_loss_type == 'listmle':
            self.ranking_loss = ListMLELoss(temperature)
        elif ranking_loss_type == 'listnet':
            self.ranking_loss = ListNetLoss(temperature)
        elif ranking_loss_type == 'bce':
            self.ranking_loss = BCERankingLoss()
        else:
            raise ValueError(f"Unknown ranking loss: {ranking_loss_type}")
        
        self.lambda_completion = lambda_completion
    
    def forward(
        self,
        results: dict,
        labels: torch.Tensor
    ) -> dict:

        losses = {}
        total_loss = torch.tensor(0.0, device=labels.device)
        
        scores = results['scores']
        ranking_loss = self.ranking_loss(scores, labels)
        losses['ranking'] = ranking_loss
        
        if not torch.isnan(ranking_loss):
            total_loss += ranking_loss
        
        if 'completion_loss' in results:
            completion_loss = results['completion_loss']
            losses['completion'] = completion_loss
            
            if not torch.isnan(completion_loss):
                total_loss += self.lambda_completion * completion_loss
        
        losses['total'] = total_loss
        
        return losses
