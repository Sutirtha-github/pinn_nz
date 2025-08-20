import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import sqrt, exp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Physical constants
hbar = torch.tensor(1.0546e-22, device=device)
kb = torch.tensor(1.3807e-23, device=device)


# Exciton operators
sx = torch.tensor([[0, -1], [-1, 0]], dtype=torch.cfloat, device=device)
S1 = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=device)

# Biexciton Operators
Mx = torch.tensor([[0, -1, 0],[-1, 0, -1],[0, -1, 0]], dtype=torch.cfloat, device=device)
M_11 = torch.tensor([[0, 0, 0],[0, 1, 0],[0, 0, 0]], dtype=torch.cfloat, device=device)
S2 = torch.tensor([[0, 0, 0],[0, 1, 0],[0, 0, 2]], dtype=torch.cfloat, device=device)


# System Hamiltonian
def rabi_const(om_b): return om_b * sqrt(torch.tensor(6.0))

def hamilton(om_b, sys='b'): 
    if sys == 'b' : return rabi_const(om_b) / 2 * Mx - 2 * om_b * M_11
    else: return  om_b / 2 * sx

def rmse(y1, y2):
    y1 = torch.tensor(y1).detach().clone()
    y2 = torch.tensor(y2).detach().clone()
    return torch.sqrt(torch.mean(torch.abs(y1-y2)**2))

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



def correlation_function(t, tp, alpha, T, omega_c):
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
    omega = torch.linspace(-15.0, 15.0, 500, device=device)
    domega = omega[1] - omega[0]
    delta_t = t - tp
    spec = 2*alpha/omega_c**2 * omega**3 * exp(-omega**2 / omega_c**2)
    nb = 1.0 / (exp(beta * omega) - 1.0)
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



def memory_kernel(t, tp, rho_tp, Hs, alpha, T, omega_c, sys='b'):
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
        sys: 'b' for bieciton (default) and 'e' for exciton model

    Returns:
        individual 2x2 (for exciton) or 3x3 (for biexciton) 
        complex-valued memory kernel matrix K(t,t').rho(t')

    '''
    if sys == 'b': S=S2
    else: S = S1
    U_tt = time_ordered_propagator(t, tp, Hs)
    c_tt = correlation_function(t, tp, alpha, T, omega_c)
    s = 1j * c_tt * (U_tt @ S @ rho_tp)
    term = s + s.conj().T
    return 1j * (S @ term - term @ S)



class ExPINN(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=4):
        super(ExPINN, self).__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 3)]
        self.model = nn.Sequential(*layers)

    def forward(self, t):
        return self.model(t)


class BiexPINN(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=4):
        super(BiexPINN, self).__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 8)]
        self.model = nn.Sequential(*layers)

    def forward(self, t):
        return self.model(t)
    


def construct_dm(output, sys='b'):
    '''
    Complete the density matrix construction from the PINN outputs

    Args:
        output : PINN predictions
        sys: 'b' for bieciton (default) and 'e' for exciton model
    '''
    if sys == 'b':
        # extract the outputs separately
        rho00 = output[:, 0:1]
        rho11 = output[:, 1:2]
        rho01_re = output[:, 2:3]
        rho01_im = output[:, 3:4]
        rho02_re = output[:, 4:5]
        rho02_im = output[:, 5:6]
        rho12_re = output[:, 6:7]
        rho12_im = output[:, 7:8]

        # create the density matrix elements
        rho01 = rho01_re + 1j * rho01_im
        rho10 = rho01_re - 1j * rho01_im
        rho02 = rho02_re + 1j * rho02_im
        rho20 = rho02_re - 1j * rho02_im
        rho12 = rho12_re + 1j * rho12_im
        rho21 = rho12_re - 1j * rho12_im
        rho22 = 1.0 - rho00 - rho11

        # construct the final 3x3 density matrix
        rho = torch.stack([
            torch.cat([rho00, rho01, rho02], dim=1),
            torch.cat([rho10, rho11, rho12], dim=1),
            torch.cat([rho20, rho21, rho22], dim=1)
        ], dim=1).requires_grad_(True)

    else:
        # extract the outputs separately
        rho00 = output[:, 0:1]
        rho01_re = output[:, 1:2]
        rho01_im = output[:, 2:3]

        # create the density matrix elements
        rho01 = rho01_re + 1j * rho01_im
        rho10 = rho01_re - 1j * rho01_im
        rho11 = 1 - rho00

        # construct the final 2x2 density matrix
        rho = torch.stack([
            torch.cat([rho00, rho01], dim=1),
            torch.cat([rho10, rho11], dim=1)
        ], dim=1).requires_grad_(True)

    return rho



def derivative_dm(rho, t, sys='b'):
    '''
    Compute time derivative of density matrix

    Args:
        rho : density matrix at time t
        t : time instance in [0,D]
        sys: 'b' for bieciton (default) and 'e' for exciton model
    '''
    rho_r, rho_i = rho.real, rho.imag

    if sys == 'b':
        rho00_r, rho01_r, rho02_r, rho10_r, rho11_r, rho12_r, rho20_r, rho21_r, rho22_r = rho_r[:, 0, 0], rho_r[:, 0, 1], rho_r[:, 0, 2], rho_r[:, 1, 0], rho_r[:, 1, 1], rho_r[:, 1, 2], rho_r[:, 2, 0], rho_r[:, 2, 1], rho_r[:, 2, 2]
        rho00_i, rho01_i, rho02_i, rho10_i, rho11_i, rho12_i, rho20_i, rho21_i, rho22_i = rho_i[:, 0, 0], rho_i[:, 0, 1], rho_i[:, 0, 2], rho_i[:, 1, 0], rho_i[:, 1, 1], rho_i[:, 1, 2], rho_i[:, 2, 0], rho_i[:, 2, 1], rho_i[:, 2, 2]

        # compute the time deivative of the real parts
        drho00_dt_r = autograd.grad(rho00_r, t, grad_outputs=torch.ones_like(rho00_r), create_graph=True)[0]
        drho01_dt_r = autograd.grad(rho01_r, t, grad_outputs=torch.ones_like(rho01_r), create_graph=True)[0]
        drho02_dt_r = autograd.grad(rho02_r, t, grad_outputs=torch.ones_like(rho02_r), create_graph=True)[0]
        drho10_dt_r = autograd.grad(rho10_r, t, grad_outputs=torch.ones_like(rho10_r), create_graph=True)[0]
        drho11_dt_r = autograd.grad(rho11_r, t, grad_outputs=torch.ones_like(rho11_r), create_graph=True)[0]
        drho12_dt_r = autograd.grad(rho12_r, t, grad_outputs=torch.ones_like(rho12_r), create_graph=True)[0]
        drho20_dt_r = autograd.grad(rho20_r, t, grad_outputs=torch.ones_like(rho20_r), create_graph=True)[0]
        drho21_dt_r = autograd.grad(rho21_r, t, grad_outputs=torch.ones_like(rho21_r), create_graph=True)[0]
        drho22_dt_r = autograd.grad(rho22_r, t, grad_outputs=torch.ones_like(rho22_r), create_graph=True)[0]

        # compute the time deivative of the imaginary parts
        drho00_dt_i = autograd.grad(rho00_i, t, grad_outputs=torch.ones_like(rho00_i), create_graph=True)[0]
        drho01_dt_i = autograd.grad(rho01_i, t, grad_outputs=torch.ones_like(rho01_i), create_graph=True)[0]
        drho02_dt_i = autograd.grad(rho02_i, t, grad_outputs=torch.ones_like(rho02_i), create_graph=True)[0]
        drho10_dt_i = autograd.grad(rho10_i, t, grad_outputs=torch.ones_like(rho10_i), create_graph=True)[0]
        drho11_dt_i = autograd.grad(rho11_i, t, grad_outputs=torch.ones_like(rho11_i), create_graph=True)[0]
        drho12_dt_i = autograd.grad(rho12_i, t, grad_outputs=torch.ones_like(rho12_i), create_graph=True)[0]
        drho20_dt_i = autograd.grad(rho20_i, t, grad_outputs=torch.ones_like(rho20_i), create_graph=True)[0]
        drho21_dt_i = autograd.grad(rho21_i, t, grad_outputs=torch.ones_like(rho21_i), create_graph=True)[0]
        drho22_dt_i = autograd.grad(rho22_i, t, grad_outputs=torch.ones_like(rho22_i), create_graph=True)[0]

        # combine the real and imaginary parts of the derivatives to form the derivative of individual entries
        drho00_dt = drho00_dt_r + 1j * drho00_dt_i
        drho01_dt = drho01_dt_r + 1j * drho01_dt_i
        drho02_dt = drho02_dt_r + 1j * drho02_dt_i
        drho10_dt = drho10_dt_r + 1j * drho10_dt_i
        drho11_dt = drho11_dt_r + 1j * drho11_dt_i
        drho12_dt = drho12_dt_r + 1j * drho12_dt_i
        drho20_dt = drho20_dt_r + 1j * drho20_dt_i
        drho21_dt = drho21_dt_r + 1j * drho21_dt_i
        drho22_dt = drho22_dt_r + 1j * drho22_dt_i

        # stack all the calculated gradients to form the final time derivative of the density matrix
        drho_dt = torch.stack([torch.cat([drho00_dt, drho01_dt, drho02_dt], dim=1),
                        torch.cat([drho10_dt, drho11_dt, drho12_dt], dim=1),
                        torch.cat([drho20_dt, drho21_dt, drho22_dt], dim=1)], dim=1).requires_grad_(True)
        
    else:

        rho00_r, rho01_r, rho10_r, rho11_r = rho_r[:, 0, 0], rho_r[:, 0, 1], rho_r[:, 1, 0], rho_r[:, 1, 1]
        rho00_i, rho01_i, rho10_i, rho11_i = rho_i[:, 0, 0], rho_i[:, 0, 1], rho_i[:, 1, 0], rho_i[:, 1, 1]

        drho00_dt_r = autograd.grad(rho00_r, t, grad_outputs=torch.ones_like(rho00_r), create_graph=True)[0]
        drho01_dt_r = autograd.grad(rho01_r, t, grad_outputs=torch.ones_like(rho01_r), create_graph=True)[0]
        drho10_dt_r = autograd.grad(rho10_r, t, grad_outputs=torch.ones_like(rho10_r), create_graph=True)[0]
        drho11_dt_r = autograd.grad(rho11_r, t, grad_outputs=torch.ones_like(rho11_r), create_graph=True)[0]

        drho00_dt_i = autograd.grad(rho00_i, t, grad_outputs=torch.ones_like(rho00_i), create_graph=True)[0]
        drho01_dt_i = autograd.grad(rho01_i, t, grad_outputs=torch.ones_like(rho01_i), create_graph=True)[0]
        drho10_dt_i = autograd.grad(rho10_i, t, grad_outputs=torch.ones_like(rho10_i), create_graph=True)[0]
        drho11_dt_i = autograd.grad(rho11_i, t, grad_outputs=torch.ones_like(rho11_i), create_graph=True)[0]

        drho00_dt = drho00_dt_r + 1j * drho00_dt_i
        drho01_dt = drho01_dt_r + 1j * drho01_dt_i
        drho10_dt = drho10_dt_r + 1j * drho10_dt_i
        drho11_dt = drho11_dt_r + 1j * drho11_dt_i

        drho_dt = torch.stack([
                        torch.cat([drho00_dt, drho01_dt], dim=1),
                        torch.cat([drho10_dt, drho11_dt], dim=1)], dim=1).requires_grad_(True)
        
    return drho_dt



def physics_loss(output, t_physics, Hs, alpha, T, omega_c, sys='b'):
    '''
    Compute the time averaged physics loss to learn the solution to the master eq.

    Args:
        output : PINN predictions for all t in t_physics
        t_physics : [0, D, t_intervals] (ps)
        Hs : System Hamiltonian
        A : system-bath coupling strength (ps/K)
        T : bath temperature (K)
        omega_c : bath cutoff frequency (1/ps)
        sys: 'b' for bieciton (default) and 'e' for exciton model

    Returns:
        Mean square of the master equation using PINN outputs (physics loss)
    
    '''
    
    rho = construct_dm(output, sys)

    # Provide grad_outputs argument to autograd.grad
    d_rho_dt = derivative_dm(rho, t_physics, sys)
    
    # Evaluate Liouvillian
    L_rho = torch.stack([calc_liouville(Hs, r) for r in rho], dim=0)
    
    # Discretize memory integral
    mem_term = torch.zeros_like(L_rho, dtype=torch.cfloat)
    
    # Accumulate autoregressive effects
    for i, ti in enumerate(t_physics):
        for j, tj in enumerate(t_physics[:i+1]):
            rho_j = rho[j]
            mem_term[i] += memory_kernel(ti, tj, rho_j, Hs, alpha, T, omega_c, sys) * (t_physics[1] - t_physics[0])

    rhs = L_rho + mem_term
    loss = torch.mean(torch.abs(d_rho_dt - rhs).squeeze()**2)
    return rho, loss


def boundary_loss(output0, sys='b'):
    '''
    Compute the boundary loss to enforce the initial conditions

    Args:
        output0: PINN prediction for t=0
        sys: 'b' for bieciton (default) and 'e' for exciton model

    Returns:
        Squared error between the actual and predicted initial boundary condition
    
    '''
    rho0 = construct_dm(output0, sys)[0]
    if sys == 'b':
        rho_true = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.cfloat, device=device)
    else: 
        rho_true = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.cfloat, device=device)

    return torch.mean(torch.abs(rho0 - rho_true).squeeze() ** 2)