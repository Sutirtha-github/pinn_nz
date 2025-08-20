import torch

def gen_noisy_data(rhos, D=1, M=50, t_intervals=1001, scale=1e-2, seed=2025):

    torch.manual_seed(seed)

    t_test = torch.linspace(0,D,t_intervals).view(-1,1)
    random_instances = torch.sort(torch.randperm(t_intervals)[:M]).values
    t_obs = t_test[random_instances].view(-1,1)
    #rho00_obs =  (torch.tensor(rho_diag_cpu[:,0])[random_instances] + 0.001*torch.randn_like(t_obs).reshape(1,-1)).flatten()
    #rho01_obs =  (torch.stack(rhos)[:,0,1][random_instances] + 0.0004*torch.randn_like(t_obs).reshape(1,-1) + 1j*0.0005*torch.randn_like(t_obs).reshape(1,-1)).flatten()
    rho_obs = torch.stack(rhos)[random_instances] + scale*torch.randn(M, 2, 2) + scale*1j*torch.randn(M, 2, 2)

    return t_obs, rho_obs