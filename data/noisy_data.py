import torch

def gen_noisy_data(rhos, D=1, M=50, t_intervals=1001, scale=1e-2, seed=2025):
    """
    Generate synthetic noisy data

    Args:

    rhos : numerically simulated density matrices
    D : pulse duration
    M : number of observational datapoints
    t_intervals : number of timepoints used in numerical simulation
    scale : noise factor used to scale normal distribution
    seed : reproducibility parameter
    """
    torch.manual_seed(seed)

    t_test = torch.linspace(0,D,t_intervals).view(-1,1)
    random_instances = torch.sort(torch.randperm(t_intervals)[:M]).values
    t_obs = t_test[random_instances].view(-1,1)
    rho_obs = torch.stack(rhos)[random_instances] + scale*torch.randn(M, 2, 2) + scale*1j*torch.randn(M, 2, 2)

    return t_obs, rho_obs
