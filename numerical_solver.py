import torch
from scipy.linalg import expm
from tqdm import tqdm
from visualizations import plot_numerical

class NZSolver:
    def __init__(self, D=0.1, steps=500, A=0.0112, omega_c=3.04, T=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.hbar = 1.0546e-22      # J*ps
        self.kb = 1.3806e-23        # J/K
        self.A = A                  # ps/K
        self.omega_c = omega_c      # 1/ps
        self.T = T                  # K
        self.beta = self.hbar / (self.kb * self.T)
        self.D = D                  # ps
        self.steps = steps
        self.dt = D / steps         # ps
        self.t = torch.linspace(0, D, steps+1, device=device, dtype=torch.float64)
        self.device = device

        # Define operators
        self.Mx = torch.tensor([[0, -1, 0], [-1, 0, -1], [0, -1, 0]], dtype=torch.complex128, device=device)
        self.Mz = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=torch.complex128, device=device)
        self.M_11 = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.complex128, device=device)
        self.S = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=torch.complex128, device=device)

        self.om_b = torch.pi / self.D
        self.Hs = self.rabi_const(self.om_b)/2 * self.Mx - 2 * self.om_b * self.M_11


    def rabi_const(self, om_b):
        return om_b * torch.sqrt(torch.tensor(6.0, device=self.device))


    def Ls_op(self, rho):
        return -1j * (self.Hs @ rho - rho @ self.Hs)


    def correlation_function(self, t, tp, omega_max=10.0, num_points=500):
        omega = torch.linspace(-omega_max, omega_max, steps=num_points, device=self.device, dtype=torch.float64)
        domega = omega[1] - omega[0]
        delta_t = t - tp

        def spectral_density(omega):
            return self.A * omega**3 * torch.exp(-omega**2 / self.omega_c**2)

        def n_beta(omega):
            exp_term = torch.exp(self.beta * omega)
            return 1.0 / (exp_term - 1.0 + 1e-10)

        spec = spectral_density(omega)
        nb = n_beta(omega)
        phase = torch.exp(1j * omega * delta_t)

        integrand = nb * spec * phase
        corr = torch.trapz(integrand, dx=domega)

        return corr


    def time_ordered_propagator(self, t, tp):
        delta_t = t - tp
        delta_t_scalar = delta_t.item() if isinstance(delta_t, torch.Tensor) else delta_t
        U = expm((-1j * self.Hs.cpu().numpy() * delta_t_scalar))  
        return torch.tensor(U, dtype=torch.complex128, device=self.device)


    def memory_kernel(self, t, tp, rho_tp):
        U_tt = self.time_ordered_propagator(t, tp)
        c_tt = self.correlation_function(t, tp)
        s = 1j * c_tt * U_tt @ self.S @ rho_tp
        term = s + s.conj().T
        return 1j * (self.S @ term - term @ self.S)


    def project_to_physical_density(self, rho):
        rho_np = rho.detach().cpu().numpy()
        eigvals, eigvecs = torch.linalg.eigh(torch.tensor(rho_np, dtype=torch.complex128))
        eigvals = torch.clamp(eigvals.real, min=0.0)
        eigvals /= eigvals.sum()
        eigvals = eigvals.to(torch.complex128)
        rho_proj = eigvecs @ torch.diag(eigvals) @ eigvecs.conj().T
        return rho_proj.to(self.device)


    def solve(self, rho_init):
        rho_t = rho_init.clone().to(self.device)
        rhos = []
        times = self.t

        for i, t in tqdm(enumerate(times), total=len(times), desc='Solving Nakajima-Zwanzig EOM'):
            integral_term = torch.zeros_like(rho_t, device=self.device, dtype=torch.complex128)

            for j, tp in enumerate(times[:i]):
                integral_term += self.memory_kernel(t, tp, rhos[j]) * self.dt

            drho_dt = self.Ls_op(rho_t) + integral_term
            rho_t = rho_t + drho_dt * self.dt

            # Enforce Hermiticity and trace preservation
            rho_t = 0.5 * (rho_t + rho_t.conj().T)
            trace = torch.trace(rho_t).abs()
            rho_t = rho_t / trace
            rho_t = self.project_to_physical_density(rho_t)

            rhos.append(rho_t.clone())

        return times, rhos


    def plot_results(self, times, rhos):
        rho_diag = torch.stack([torch.real(torch.diag(rho)) for rho in rhos])
        total_population = torch.sum(rho_diag, dim=1)
        times_cpu = times.detach().cpu().numpy()
        rho_diag_cpu = rho_diag.detach().cpu().numpy()
        total_cpu = total_population.detach().cpu().numpy()
        plot_numerical(times_cpu, rho_diag_cpu, total_cpu, self.D)