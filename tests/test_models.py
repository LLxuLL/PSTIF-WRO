import torch
import pytest
from models.pstif_wro import PSTIFWRO, PSTIFWROConfig
from models.if_measure_embedding import IFMeasureEmbedding
from models.pw_gcn import PWGCN
from models.wasserstein_critic import WassersteinCritic


def test_if_measure_embedding():
    batch_size = 4
    seq_len = 10
    input_dim = 5
    
    embedding = IFMeasureEmbedding(input_dim, hidden_dim=32)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    measures, components = embedding(x, return_components=True)
    
    assert measures.shape == (batch_size, seq_len, 3)
    assert torch.allclose(measures.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)


def test_wasserstein_critic():
    batch_size = 4
    input_dim = 3
    
    critic = WassersteinCritic(input_dim, hidden_dims=[32, 16])
    x = torch.randn(batch_size, input_dim)
    
    scores = critic(x)
    
    assert scores.shape == (batch_size, 1)


def test_pw_gcn():
    num_nodes = 10
    in_channels = 3
    num_partitions = 3
    
    pw_gcn = PWGCN(
        in_channels=in_channels,
        hidden_channels=[16, 16],
        out_channels=3,
        num_partitions=num_partitions
    )
    
    x = torch.randn(num_nodes, in_channels)
    partition_ids = torch.randint(0, num_partitions, (num_nodes,))
    
    out = pw_gcn(x, partition_ids)
    
    assert out.shape == (num_nodes, 3)


def test_pstif_wro():
    batch_size = 2
    num_attributes = 5
    input_dim = 1
    num_partitions = 3
    
    model = PSTIFWRO(
        input_dim=input_dim,
        num_attributes=num_attributes,
        num_partitions=num_partitions,
        hidden_dim=16
    )
    
    x = torch.randn(batch_size, num_attributes, input_dim)
    partition_ids = torch.randint(0, num_partitions, (batch_size, num_attributes))
    missing_mask = torch.ones(batch_size, num_attributes, dtype=torch.bool)
    
    results = model(x, partition_ids, missing_mask=missing_mask)
    
    assert 'scores' in results
    assert results['scores'].shape == (batch_size,)
    assert 'measures' in results
    assert results['measures'].shape == (batch_size, num_attributes, 3)


if __name__ == '__main__':
    test_if_measure_embedding()
    test_wasserstein_critic()
    test_pw_gcn()
    test_pstif_wro()
    print("All tests passed!")
