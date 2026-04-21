import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class SimpleMessagePassing(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super(SimpleMessagePassing, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lin.weight, gain=0.5)
        nn.init.zeros_(self.lin.bias)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape

        h = self.lin(x)

        degree = adjacency.sum(dim=-1, keepdim=True) + 1e-8  # (batch, num_nodes, 1)
        adj_norm = adjacency / degree  # (batch, num_nodes, num_nodes)

        # (batch, num_nodes, num_nodes) @ (batch, num_nodes, out_channels)
        # -> (batch, num_nodes, out_channels)
        h = torch.bmm(adj_norm, h)

        h = h.view(-1, self.out_channels)  # (batch*num_nodes, out_channels)
        h = self.bn(h)
        h = h.view(batch_size, num_nodes, self.out_channels)

        h = F.relu(h)
        h = self.dropout(h)

        return h


class PWGCN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        num_partitions: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_temporal: bool = True
    ):
        super(PWGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_partitions = num_partitions
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_temporal = use_temporal

        self.convs = nn.ModuleList()

        self.convs.append(
            SimpleMessagePassing(in_channels, hidden_channels[0], dropout)
        )

        for i in range(len(hidden_channels) - 1):
            self.convs.append(
                SimpleMessagePassing(
                    hidden_channels[i],
                    hidden_channels[i + 1],
                    dropout
                )
            )

        self.convs.append(
            SimpleMessagePassing(
                hidden_channels[-1],
                out_channels,
                dropout
            )
        )

        self.partition_correlation = nn.Parameter(
            torch.eye(num_partitions) * 0.5 + 0.5
        )

        if use_temporal:
            self.temporal_decay = nn.Parameter(torch.tensor(0.01))

    def _build_partition_adjacency_fast(
        self,
        partition_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size, num_nodes = partition_ids.shape
        device = partition_ids.device

        # (batch, num_nodes, 1) vs (batch, 1, num_nodes)
        p_i = partition_ids.unsqueeze(-1)  # (batch, num_nodes, 1)
        p_j = partition_ids.unsqueeze(1)   # (batch, 1, num_nodes)

        # partition_correlation: (num_partitions, num_partitions)
        # p_i, p_j: (batch, num_nodes, num_nodes)
        weights = self.partition_correlation[p_i, p_j]  # (batch, num_nodes, num_nodes)

        if self.use_temporal and timestamps is not None:
            # (batch, num_nodes, 1) - (batch, 1, num_nodes) -> (batch, num_nodes, num_nodes)
            time_diff = torch.abs(timestamps.unsqueeze(-1) - timestamps.unsqueeze(1))
            time_weight = torch.exp(-self.temporal_decay * time_diff)
            weights = weights * time_weight

        eye = torch.eye(num_nodes, device=device).unsqueeze(0)  # (1, num_nodes, num_nodes)
        adjacency = weights + eye

        return adjacency

    def forward(
        self,
        x: torch.Tensor,
        partition_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        adjacency = self._build_partition_adjacency_fast(partition_ids, timestamps)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adjacency)

        x = self.convs[-1](x, adjacency)

        return x


class TemporalPartitionGraph(nn.Module):

    def __init__(
        self,
        num_partitions: int,
        temporal_decay: float = 0.01
    ):
        super(TemporalPartitionGraph, self).__init__()

        self.num_partitions = num_partitions

        self.partition_correlation = nn.Parameter(
            torch.eye(num_partitions) + 0.1
        )
        self.temporal_decay = nn.Parameter(torch.tensor(temporal_decay))

    def build_graph(
        self,
        features: torch.Tensor,
        partition_ids: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:

        batch_size, num_nodes, _ = features.shape
        device = features.device

        p_i = partition_ids.unsqueeze(-1)  # (batch, num_nodes, 1)
        p_j = partition_ids.unsqueeze(1)   # (batch, 1, num_nodes)

        weights = self.partition_correlation[p_i, p_j]  # (batch, num_nodes, num_nodes)

        time_diff = torch.abs(timestamps.unsqueeze(-1) - timestamps.unsqueeze(1))
        time_weight = torch.exp(-self.temporal_decay * time_diff)
        weights = weights * time_weight

        eye = torch.eye(num_nodes, device=device).unsqueeze(0)
        adjacency = weights + eye

        return adjacency