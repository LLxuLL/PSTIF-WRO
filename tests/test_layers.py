import torch
import pytest
from layers.sinkhorn import SinkhornDistance, WassersteinBarycenter
from layers.gradient_penalty import GradientPenalty


def test_sinkhorn_distance():
    batch_size = 4
    n = 10
    m = 8
    d = 3
    
    sinkhorn = SinkhornDistance(blur=0.05, max_iter=50)
    
    source = torch.randn(batch_size, n, d)
    target = torch.randn(batch_size, m, d)
    
    distance, P = sinkhorn(source, target)
    
    assert distance.shape == (batch_size,)
    assert P.shape == (batch_size, n, m)
    assert torch.all(distance >= 0)


def test_wasserstein_barycenter():
    batch_size = 4
    num_measures = 5
    measure_dim = 3
    
    barycenter = WassersteinBarycenter(blur=0.05, barycenter_iter=5)
    
    measures = torch.rand(batch_size, num_measures, measure_dim)
    measures = measures / measures.sum(dim=-1, keepdim=True)
    
    bary = barycenter(measures)
    
    assert bary.shape == (batch_size, measure_dim)
    assert torch.allclose(bary.sum(dim=-1), torch.ones(batch_size), atol=1e-4)


def test_gradient_penalty():
    batch_size = 4
    input_dim = 10
    
    critic = torch.nn.Linear(input_dim, 1)
    gp = GradientPenalty(lambda_gp=10.0)
    
    real_samples = torch.randn(batch_size, input_dim)
    fake_samples = torch.randn(batch_size, input_dim)
    
    penalty = gp(critic, real_samples, fake_samples)
    
    assert penalty.shape == ()
    assert penalty.item() >= 0


if __name__ == '__main__':
    test_sinkhorn_distance()
    test_wasserstein_barycenter()
    test_gradient_penalty()
    print("All layer tests passed!")
