import torch
from torch.autograd.functional import jacobian
from visualizations import plot_sensitivity


hbar = 1.0546e-22
k_B = 1.3806e-23
# Grid
t_vals = torch.linspace(0, 5, 100)
omega_vals = torch.linspace(-10.0, 10.0, 500)
domega = omega_vals[1] - omega_vals[0]

t_grid, tp_grid = torch.meshgrid(t_vals, t_vals, indexing='ij')
tau_grid = t_grid - tp_grid  

# Parameters as tensors
alpha_val = 0.126
omega_c_val = 3.04
T_val = 10.0

# Spectral density
def J(omega, alpha, omega_c):
    return (2 * alpha / omega_c**2) * omega**3 * torch.exp(-omega**2 / omega_c**2)

# Bose-Einstein occupation
def n_BE(omega, T):
    beta = hbar / (k_B*T)
    x = beta * omega
    return 1.0 / (torch.exp(x) - 1.0)


# Correlation function as flattened vector
def c_tt_param(params):
    alpha, omega_c, T = params
    omega = omega_vals.view(1, 1, -1)
    tau = tau_grid.unsqueeze(-1)
    Jw = J(omega, alpha, omega_c)
    nw = n_BE(omega, T)
    integrand = Jw * nw * torch.cos(omega * tau)
    c_matrix = torch.sum(integrand, dim=-1) * domega
    return c_matrix.view(-1)  

params = torch.tensor([alpha_val, omega_c_val, T_val], requires_grad=True)

# Compute Jacobian: shape = [M*M, 3]
JAC = jacobian(c_tt_param, params).T  
dc_dalpha = JAC[0].view(t_grid.shape)
dc_domega_c = JAC[1].view(t_grid.shape)

plot_sensitivity(dc_dalpha, dc_domega_c)