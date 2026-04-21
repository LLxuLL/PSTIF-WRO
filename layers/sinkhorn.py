import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinkhornDistance(nn.Module):
    
    def __init__(
        self,
        p: int = 2,
        blur: float = 0.05,
        max_iter: int = 20,
        threshold: float = 1e-3,
    ):
        super(SinkhornDistance, self).__init__()
        
        self.p = p
        self.blur = blur
        self.max_iter = max_iter
        self.threshold = threshold
    
    def _cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        x_norm = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm = (y ** 2).sum(dim=-1, keepdim=True)
        
        # C[i,j] = ||x_i - y_j||^2
        C = x_norm + y_norm.transpose(-2, -1) - 2 * torch.matmul(x, y.transpose(-2, -1))
        C = C.clamp(min=0.0)
        
        return C
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_weights: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size = source.shape[0]
        
        if source.dim() == 2:
            diff = source - target
            distance = torch.norm(diff, p=2, dim=-1)
            return distance
        
        C = self._cost_matrix(source, target)
        

        min_cost = C.min(dim=-1)[0].mean(dim=-1)
        
        return min_cost


class StableSinkhornDistance(nn.Module):
    
    def __init__(
        self,
        p: int = 2,
        blur: float = 0.05,
        max_iter: int = 10,
    ):
        super(StableSinkhornDistance, self).__init__()
        
        self.p = p
        self.eps = blur
        self.max_iter = max_iter
    
    def forward(self, mu: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:

        batch_size, n = mu.shape
        m = nu.shape[1]
        
        mu = F.softmax(mu, dim=-1)
        nu = F.softmax(nu, dim=-1)
        

        mu_mean = mu.sum(dim=-1)
        nu_mean = nu.sum(dim=-1)
        mean_diff = (mu_mean - nu_mean) ** 2

        distance = torch.sqrt(mean_diff + 1e-8)
        
        return distance


class SinkhornAttention(nn.Module):
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        temperature: float = 0.1
    ):
        super(SinkhornAttention, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, _ = query.shape
        
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).expand(-1, seq_len, -1), -1e9)
        
        attn_weights = F.softmax(scores / self.temperature, dim=-1)
        
        output = torch.bmm(attn_weights, V)
        output = self.out_proj(output)
        
        return output


class WassersteinBarycenter(nn.Module):

    
    def __init__(self, measure_dim: int = 3):
        super(WassersteinBarycenter, self).__init__()
        
        self.measure_dim = measure_dim
    
    def forward(
        self,
        measures: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size, num_measures, measure_dim = measures.shape
        
        if weights is None:
            weights = torch.ones(batch_size, num_measures, device=measures.device) / num_measures
        
        weights = weights.unsqueeze(-1)  # (batch, num_measures, 1)
        barycenter = (measures * weights).sum(dim=1)  # (batch, measure_dim)
        
        barycenter = F.softmax(barycenter, dim=-1)
        
        return barycenter
