import torch
import torch.nn as nn
from torch import pi, sqrt, exp
from visualizations import plot_midtraining



# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Physical constants
hbar = torch.tensor(1.0546e-22, device=device)
kb = torch.tensor(1.3807e-23, device=device)


# Operators
Mx = torch.tensor([[0,-1,0],[-1,0,-1],[0,-1,0]], dtype=torch.complex64, device=device)
Mz = torch.tensor([[1,0,0],[0,0,0],[0,0,-1]], dtype=torch.complex64, device=device)
M_11 = torch.tensor([[0,0,0],[0,1,0],[0,0,0]], dtype=torch.complex64, device=device)
S = torch.tensor([[0,0,0],[0,1,0],[0,0,2]], dtype=torch.complex64, device=device)


# System Hamiltonian
def rabi_const(om_b): return om_b * sqrt(torch.tensor(6.0))
def hamilton(om_b): return rabi_const(om_b) / 2 * Mx - 2 * om_b * M_11



class WavefunctionNN(nn.Module):
    '''
    PINN architecture - fully connected NN

    Args:
        n_hidden: # units per hidden layer
        n_layers: # hidden layers

    Input:
        t: time instance
        
    Output:
        real and imaginary parts of probability amplitudes c0(t), c1(t), c2(t)

    '''
    def __init__(self, n_hidden=64, n_layers=4):
        super().__init__()
        activation = nn.Tanh
        self.input_layer = nn.Sequential(nn.Linear(1, n_hidden), activation())
        self.hidden_layers = nn.Sequential(*[nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(n_layers - 1)])
        self.output_layer = nn.Linear(n_hidden, 6)

    def forward(self, t):
        x = self.input_layer(t)
        x = self.hidden_layers(x)
        return self.output_layer(x)



def calc_liouville(Hs, rho):
    '''
    Liouvillian superoperator

    Args:
        Hs: system hamiltonian
        rho: density matrix

    Returns:
        complex-valued 3x3 Liouvillian superoperator matrix 
    '''
    return -1j * (Hs @ rho - rho @ Hs)



def correlation_function(t, tp, A, T, omega_c):
    '''
    Compute correlation function c(t,t').

    Args:
        t: current time instance (ps)
        tp: previous time instance (ps)
        A: system-bath coupling strength (ps/K)
        T : bath temperature (K)
        omega_c: cutoff frequency (1/ps)

    Returns:
        complex-valued correlation function at t, t'

    '''

    beta = hbar / (kb * T)
    omega = torch.linspace(-10.0, 10.0, 250, device=device)
    domega = omega[1] - omega[0]
    delta_t = t - tp
    spec = A * omega**3 * exp(-omega**2 / omega_c**2)
    nb = 1.0 / (exp(beta * omega) - 1.0 + 1e-10)
    integrand = nb * spec * exp(1j * omega * delta_t)
    return torch.sum(integrand, dim=-1) * domega



def time_ordered_propagator(t, tp, Hs):
    '''
    Compute time-ordered unitary propagator U(t, t')

    Args:
        t: current time instance (ps)
        tp: previous time instance (ps)
        Hs: system hamiltonian

    Returns:
        U(t,t')
    '''
    return torch.matrix_exp(-1j * Hs * (t - tp))



def memory_kernel(t, tp, rho_tp, Hs, A, T, omega_c):
    '''
    Memory kernel term K(t,t') rho(t')

    Args:
        t: current time instance (ps)
        tp: previous time instance (ps)
        rho_tp: density matrix at time t'
        Hs: system hamiltonian
        A: system-bath coupling strength (ps/K)
        T : bath temperature (K)
        omega_c: cutoff frequency (1/ps)

    Returns:
        individual 3x3 complex-valued memory kernel matrix K(t,t').rho(t')

    '''
    U_tt = time_ordered_propagator(t, tp, Hs)
    c_tt = correlation_function(t, tp, A, T, omega_c)
    s = 1j * c_tt * (U_tt @ S @ rho_tp)
    term = s + s.conj().T
    return 1j * (S @ term - term @ S)


# Training 
def train_nz_pinn(t, rho_diag, D=0.1, t_intervals = 150, A=0.0112, T=100, omega_c=3.04,
                  n_layers=4, n_hidden=32, epochs=3501, lr=1e-2, plot_interval=500):

    torch.manual_seed(2025)

    dt = D / t_intervals
    t_test = torch.linspace(0, D, t_intervals, device=device).view(-1, 1)

    om_b = pi / D
    Hs = hamilton(om_b)

    pinn = WavefunctionNN(n_hidden, n_layers).to(device)
    t_boundary = torch.tensor([[0.0]], device=device, requires_grad=True)
    t_physics = torch.linspace(0, D, t_intervals, device=device).view(-1, 1).requires_grad_(True)

    optimiser = torch.optim.Adam(pinn.parameters(), lr=lr)

    for i in range(epochs):

        optimiser.zero_grad()

        rho0_pred = pinn(t_boundary)
        loss_b = torch.mean((rho0_pred[:, 0] - 1.0) ** 2 + torch.sum(rho0_pred[:, 1:] ** 2, dim=1))

        psi_flat = pinn(t_physics)
        psi_r = psi_flat[:, :3].view(-1, 3, 1)
        psi_i = psi_flat[:, 3:].view(-1, 3, 1)
        psi = torch.complex(psi_r, psi_i)
        rho = psi @ psi.conj().transpose(-1, -2)

        grads = [torch.autograd.grad(psi_r[:, j], t_physics, torch.ones_like(psi_r[:, j]), create_graph=True)[0] for j in range(3)]
        grads += [torch.autograd.grad(psi_i[:, j], t_physics, torch.ones_like(psi_i[:, j]), create_graph=True)[0] for j in range(3)]
        d_psi_dt = torch.complex(torch.stack(grads[:3], dim=1), torch.stack(grads[3:], dim=1)).view(-1, 3, 1)

        d_rho_dt = psi @ d_psi_dt.conj().transpose(-1, -2) + d_psi_dt @ psi.conj().transpose(-1, -2)
        first_terms = calc_liouville(Hs, rho)

        second_terms = []
        for j in range(len(t_physics)):
            kernel_term = memory_kernel(t_physics[j], t_physics[0], rho[0], Hs, A, T, omega_c)
            for k in range(1, j + 1):
                kernel_term += memory_kernel(t_physics[j], t_physics[k], rho[k], Hs, A, T, omega_c)
            second_terms.append(kernel_term)
        second_terms = dt * torch.stack(second_terms)

        loss_p = torch.mean(torch.abs(d_rho_dt - first_terms - second_terms) ** 2)
        loss_norm = torch.mean((torch.real(torch.vmap(torch.trace)(rho)) - 1.0) ** 2)
        loss = loss_b + 0.001 * loss_p + loss_norm
        
        loss.backward()
        optimiser.step()

        if i % plot_interval == 0:
            print(f"Epoch: {i}  |  Loss = {loss.item():.4e}")
            plot_midtraining(t, rho_diag, t_test, rho, D)

    return pinn
