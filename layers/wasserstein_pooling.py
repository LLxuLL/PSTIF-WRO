import torch
import torch.nn as nn
import torch.nn.functional as F


class WassersteinBarycenterPooling(nn.Module):

    
    def __init__(self, measure_dim: int = 3):
        super(WassersteinBarycenterPooling, self).__init__()
        
        self.measure_dim = measure_dim
    
    def forward(
        self,
        measures: torch.Tensor,
        weights: torch.Tensor = None
    ) -> torch.Tensor:

        batch_size, num_measures, _ = measures.shape
        
        if weights is None:
            weights = torch.ones(batch_size, num_measures, device=measures.device)
            weights = weights / num_measures
        
        weights = weights.unsqueeze(-1)  # (batch, num_measures, 1)
        barycenter = (measures * weights).sum(dim=1)  # (batch, measure_dim)
        
        barycenter = F.softmax(barycenter, dim=-1)
        
        return barycenter


class AttentionalWassersteinPooling(nn.Module):

    
    def __init__(self, measure_dim: int = 3, hidden_dim: int = 64):
        super(AttentionalWassersteinPooling, self).__init__()
        
        self.measure_dim = measure_dim

        self.attention = nn.Sequential(
            nn.Linear(measure_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, measures: torch.Tensor) -> torch.Tensor:

        attn_scores = self.attention(measures).squeeze(-1)  # (batch, num_measures)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, num_measures)
        
        pooled = torch.bmm(
            attn_weights.unsqueeze(1),
            measures
        ).squeeze(1)  # (batch, measure_dim)
        
        pooled = F.softmax(pooled, dim=-1)
        
        return pooled
