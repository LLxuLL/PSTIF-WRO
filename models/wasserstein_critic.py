import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class WassersteinCritic(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        use_gradient_penalty: bool = False,
        lambda_gp: float = 10.0
    ):
        super(WassersteinCritic, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_gradient_penalty = use_gradient_penalty
        self.lambda_gp = lambda_gp
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return self.network(x)


class RobustWassersteinScore(nn.Module):
    
    def __init__(
        self,
        measure_dim: int = 3,
        eps: float = 0.1,
        critic_hidden_dims: list = [64, 32],
        use_gradient_penalty: bool = False
    ):
        super(RobustWassersteinScore, self).__init__()
        
        self.measure_dim = measure_dim
        self.eps = eps
        
        # Critic Net
        self.critic = WassersteinCritic(
            input_dim=measure_dim,
            hidden_dims=critic_hidden_dims,
            use_gradient_penalty=use_gradient_penalty
        )
    
    def forward(
        self,
        measure: torch.Tensor,
        compute_robust: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        score = self.critic(measure)
        
        if not compute_robust:
            return score, None
        
        noise = torch.randn_like(measure) * self.eps * 0.1
        measure_perturbed = measure + noise
        measure_perturbed = F.softmax(measure_perturbed, dim=-1)
        
        robust_score = self.critic(measure_perturbed)
        
        return score, robust_score


class MultiCriticEnsemble(nn.Module):
    
    def __init__(
        self,
        num_critics: int = 1,
        input_dim: int = 3,
        hidden_dims: list = [64, 32],
        **kwargs
    ):
        super(MultiCriticEnsemble, self).__init__()
        
        self.num_critics = num_critics
        
        self.critics = nn.ModuleList([
            WassersteinCritic(input_dim, hidden_dims, **kwargs)
            for _ in range(num_critics)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        scores = [critic(x) for critic in self.critics]
        return torch.stack(scores, dim=0).mean(dim=0)
