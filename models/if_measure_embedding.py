import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class IFMeasureEmbedding(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        measure_dim: int = 3,
        dropout: float = 0.1
    ):
        super(IFMeasureEmbedding, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.measure_dim = measure_dim

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.measure_generator = nn.Linear(hidden_dim, measure_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:

        original_shape = x.shape

        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        elif x.dim() == 3:
            squeeze_output = False
            if x.shape[-1] != self.input_dim:
                if x.shape[1] == self.input_dim:
                    x = x.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}, shape: {original_shape}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch. Expected last dim={self.input_dim}, got shape={x.shape}")

        batch_size, seq_len, _ = x.shape

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        features = self.feature_encoder(x)
        measure_logits = self.measure_generator(features)
        measures = F.softmax(measure_logits, dim=-1)

        if mask is not None:
            if mask.dim() == 2:
                mask_expanded = mask.unsqueeze(-1).float()
            else:
                mask_expanded = mask.float().unsqueeze(-1) if mask.dim() == 1 else mask.float()
            uniform_measure = torch.ones_like(measures) / self.measure_dim
            measures = measures * mask_expanded + uniform_measure * (1 - mask_expanded)
            measures = measures / (measures.sum(dim=-1, keepdim=True) + 1e-8)

        if squeeze_output:
            measures = measures.squeeze(1)

        if return_components:
            mu = measures[..., 0]
            nu = measures[..., 1]
            pi = measures[..., 2]
            return measures, (mu, nu, pi)
        else:
            return measures, None


class IFMeasureBatchEmbedding(nn.Module):

    def __init__(
        self,
        num_attributes: int,
        input_dim: int,
        hidden_dim: int = 64,
        measure_dim: int = 16,
        share_weights: bool = True,
        dropout: float = 0.1
    ):
        super(IFMeasureBatchEmbedding, self).__init__()

        self.num_attributes = num_attributes
        self.input_dim = input_dim
        self.share_weights = share_weights

        if share_weights:
            self.shared_embedding = IFMeasureEmbedding(
                input_dim, hidden_dim, measure_dim, dropout
            )
        else:
            self.embeddings = nn.ModuleList([
                IFMeasureEmbedding(input_dim, hidden_dim, measure_dim, dropout)
                for _ in range(num_attributes)
            ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D with shape {x.shape}")

        if x.shape[1] != self.num_attributes:
            raise ValueError(f"Number of attributes mismatch. Expected {self.num_attributes}, got {x.shape[1]}")

        if self.share_weights:

            result = self.shared_embedding(x, mask, return_components=False)
            if not isinstance(result, tuple):
                measures = result
                components = None
            else:
                measures, components = result
        else:
            measures_list = []
            for i in range(self.num_attributes):
                attr_x = x[:, i, :]
                attr_mask = mask[:, i] if mask is not None else None
                result = self.embeddings[i](attr_x, attr_mask, False)

                if isinstance(result, tuple):
                    attr_measure = result[0]
                else:
                    attr_measure = result
                measures_list.append(attr_measure)

            measures = torch.stack(measures_list, dim=1)
            components = None

        if return_components and components is not None:
            return measures, components

        return measures, None