import torch
import torch.nn as nn


class GradientPenalty(nn.Module):
    
    def __init__(self, lambda_gp: float = 10.0):
        super(GradientPenalty, self).__init__()
        self.lambda_gp = lambda_gp
    
    def forward(
        self,
        critic: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor
    ) -> torch.Tensor:

        batch_size = real_samples.shape[0]
        device = real_samples.device
        
        alpha = torch.rand(batch_size, 1, device=device)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        d_interpolates = critic(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients_norm = gradients.norm(2, dim=1)
        
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * self.lambda_gp
        
        return gradient_penalty


class SpectralNormalization(nn.Module):

    
    def __init__(self, module: nn.Module, n_power_iterations: int = 1):
        super(SpectralNormalization, self).__init__()
        self.module = nn.utils.spectral_norm(module, n_power_iterations=n_power_iterations)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
