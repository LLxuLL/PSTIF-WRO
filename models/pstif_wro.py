import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from models.if_measure_embedding import IFMeasureBatchEmbedding
from models.pw_gcn import PWGCN
from models.contrastive_completion import AdaptiveContrastiveCompletion
from layers.wasserstein_pooling import WassersteinBarycenterPooling


class WassersteinCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout: float = 0.1
    ):
        super(WassersteinCritic, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
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


class PSTIFWRO(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_attributes: int,
        num_partitions: int,
        hidden_dim: int = 64,
        measure_dim: int = 3,
        use_contrastive: bool = False,
        use_robust: bool = False,
        dropout: float = 0.1,
        config: Optional[Dict] = None
    ):
        super(PSTIFWRO, self).__init__()

        self.input_dim = input_dim
        self.num_attributes = num_attributes
        self.num_partitions = num_partitions
        self.hidden_dim = hidden_dim
        self.measure_dim = measure_dim
        self.use_contrastive = use_contrastive
        self.use_robust = use_robust
        self.dropout = dropout

        config = config or {}

        self.if_embedding = IFMeasureBatchEmbedding(
            num_attributes=num_attributes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            measure_dim=measure_dim,
            share_weights=True,
            dropout=dropout
        )

        if use_contrastive:
            self.completion = AdaptiveContrastiveCompletion(
                measure_dim=measure_dim,
                hidden_dim=hidden_dim
            )

        # PW-GCN
        self.pw_gcn = PWGCN(
            in_channels=measure_dim,
            hidden_channels=[hidden_dim, hidden_dim],
            out_channels=measure_dim,
            num_partitions=num_partitions,
            num_layers=2,
            dropout=dropout,
            use_temporal=config.get('use_temporal', True)
        )

        self.barycenter_pooling = WassersteinBarycenterPooling(
            measure_dim=measure_dim
        )

        # Wasserstein Critic
        self.critic = WassersteinCritic(
            input_dim=measure_dim,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        partition_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        results = {}

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size = x.shape[0]

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        embedding_result = self.if_embedding(x, missing_mask, return_components=False)
        if isinstance(embedding_result, tuple):
            measures = embedding_result[0]
        else:
            measures = embedding_result

        results['measures'] = measures

        completion_loss = torch.tensor(0.0, device=x.device)

        if self.use_contrastive and missing_mask is not None:
            completion_result = self.completion(measures, missing_mask, None)
            if isinstance(completion_result, tuple) and len(completion_result) == 2:
                measures, completion_loss = completion_result
            else:
                measures = completion_result[0] if isinstance(completion_result, tuple) else completion_result
                completion_loss = torch.tensor(0.0, device=x.device)

            results['completion_loss'] = completion_loss

        aggregated_measures = self.pw_gcn(
            measures,
            partition_ids,
            timestamps
        )
        results['aggregated_measures'] = aggregated_measures

        global_measures = self.barycenter_pooling(aggregated_measures)
        results['global_measures'] = global_measures

        scores = self.critic(global_measures)
        results['scores'] = scores.squeeze(-1)

        return results

    def compute_loss(
        self,
        results: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        loss_weights = loss_weights or {
            'ranking': 1.0,
            'completion': 0.01,
        }

        losses = {}
        total_loss = torch.tensor(0.0, device=labels.device)

        scores = results['scores']

        labels_normalized = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)

        ranking_loss = F.binary_cross_entropy_with_logits(
            scores, labels_normalized.float()
        )
        losses['ranking'] = ranking_loss
        total_loss += loss_weights['ranking'] * ranking_loss

        if 'completion_loss' in results:
            completion_loss = results['completion_loss']
            losses['completion'] = completion_loss
            total_loss += loss_weights['completion'] * completion_loss

        losses['total'] = total_loss

        return losses

    def predict(self, x, partition_ids, timestamps=None, spatial_coords=None, missing_mask=None):
        self.eval()
        with torch.no_grad():
            results = self.forward(x, partition_ids, timestamps, spatial_coords, missing_mask)
            return torch.sigmoid(results['scores'])


class PSTIFWROConfig:
    DEFAULT_CONFIG = {
        'hidden_dim': 64,
        'dropout': 0.1,
        'use_temporal': True,
    }

    @classmethod
    def get_config(cls, overrides: Optional[Dict] = None) -> Dict:
        config = cls.DEFAULT_CONFIG.copy()
        if overrides:
            config.update(overrides)
        return config