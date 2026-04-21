import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ContrastiveCompletion(nn.Module):

    def __init__(
        self,
        measure_dim: int = 3,
        hidden_dim: int = 64,
        temperature: float = 0.1
    ):
        super(ContrastiveCompletion, self).__init__()

        self.measure_dim = measure_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(measure_dim, hidden_dim),
            nn.ReLU(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, measure_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, measures: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(measures), dim=-1)

    def forward(
        self,
        measures: torch.Tensor,
        missing_mask: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_attrs, _ = measures.shape
        device = measures.device

        completed_measures = measures.clone()

        for b in range(batch_size):
            # 获取存在的属性
            context_mask = missing_mask[b] if missing_mask is not None else torch.ones(num_attrs, dtype=torch.bool, device=device)

            if context_mask.sum() == 0:
                completed_measures[b] = torch.ones(
                    num_attrs, self.measure_dim, device=device
                ) / self.measure_dim
                continue

            context_measures = measures[b, context_mask]
            context_mean = context_measures.mean(dim=0)

            for attr_idx in range(num_attrs):
                if not context_mask[attr_idx]:
                    noise = torch.randn_like(context_mean) * 0.01
                    completed_measures[b, attr_idx] = F.softmax(
                        context_mean + noise, dim=-1
                    )

        loss = F.mse_loss(completed_measures, measures.detach())

        return completed_measures, loss


class AdaptiveContrastiveCompletion(nn.Module):

    def __init__(
        self,
        measure_dim: int = 3,
        hidden_dim: int = 64,
        temperature: float = 0.1
    ):
        super(AdaptiveContrastiveCompletion, self).__init__()

        self.base_completion = ContrastiveCompletion(
            measure_dim, hidden_dim, temperature
        )

    def forward(
        self,
        measures: torch.Tensor,
        missing_mask: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        result = self.base_completion(measures, missing_mask, adjacency)

        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(f"ContrastiveCompletion returned {type(result)}, expected tuple of 2")
        return result[0], result[1]