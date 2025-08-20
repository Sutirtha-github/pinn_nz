import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def plot_numerical(times_cpu, rho_diag_cpu, total_cpu, D, sys='b'):
    '''
    Plot the dynamics obtained by numerically solving the NZ eq.

    Args:
        times_cpu: list of time points (stored in cpu) used from 0 to pulse duration
        rho_diag_cpu: population of ground state, exciton and biexciton (diagonal elements of density matrix) for all time points (stored in cpu)
        total_cpu: sum of individual populations i.e. trace of density matrix for all times (stored in cpu)
        D: pulse duration
        sys: 'b' for bieciton (default) and 'e' for exciton model
    '''
    plt.figure(figsize=(5, 4))
    plt.rcParams['font.size'] = 12.5

    if sys == 'b':
        plt.plot(times_cpu, rho_diag_cpu[:, 0], label=r'$\rho_{00}$', color="red", linewidth=2)
        plt.plot(times_cpu, rho_diag_cpu[:, 1], label=r'$\rho_{11}$', color="green", linewidth=2)
        plt.plot(times_cpu, rho_diag_cpu[:, 2], label=r'$\rho_{22}$', color="blue", linewidth=2)
        plt.plot(times_cpu, total_cpu, '--', label=r'Tr($\rho$)', color='black', linewidth=1.5)
        plt.xlabel('Time (ps)')
        plt.ylabel('Population')
        plt.title('Numerical NZ solution of Biexciton')
        plt.xlim(0, D)
        plt.ylim(0, 1.1)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    else:
        plt.plot(times_cpu, rho_diag_cpu[:, 0], label=r'$\rho_{00}$', color="tab:red",linewidth=2)
        plt.plot(times_cpu, rho_diag_cpu[:, 1], label=r'$\rho_{11}$', color="tab:blue", linewidth=2)
        plt.xlabel('Time (ps)')
        plt.ylabel('Population')
        plt.title('Numerical NZ solution of Exciton')
        plt.xlim(0, D)
        plt.ylim(0, 1)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()


def plot_midtrain_sim(t_num, rho_diag_num, t_test, rho_pred, D, sys='b'):
    '''
    Plot the learnt dynamics along with the numerically obtained dynamics at different epochs of the training

    Args:
        t_num: list of time points [0,D] used in numerical solution
        rho_diag_num: diagonal elements obtained from numerical solutions for all time points
        t_test: time points used during inference
        rho_pred: predicted density matrix for all time instances in t_intervals
        D: pulse duration
        sys: 'b' for bieciton (default) and 'e' for exciton model

    '''
    
    plt.rcParams['font.size'] = 12.5

    rho_pred = rho_pred.detach()

    # Interpolation for smoothing predictions
    t_test_interp = torch.linspace(t_test[:,0].min(), t_test[:,0].max(), 100)
    rho00_spline = make_interp_spline(t_test[:,0], torch.abs(rho_pred[:,0,0]))
    rho11_spline = make_interp_spline(t_test[:,0], torch.abs(rho_pred[:,1,1]))

    if sys == 'b':
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        rho22_spline = make_interp_spline(t_test[:,0], torch.abs(rho_pred[:,2,2]))

        axes[0].plot(t_num, rho_diag_num[:,0], label="Numerical solution", color="black", linewidth=2)
        axes[0].plot(t_test_interp, rho00_spline(t_test_interp), label="PINN solution", color="tab:red", linewidth=2)
        axes[0].set_xlabel("t (ps)")
        axes[0].set_ylabel(r"$\rho_{00}(t)$")
        axes[0].set_xlim(0,D)
        axes[0].set_ylim(0,1)
        axes[0].legend(fontsize=11)

        
        axes[1].plot(t_num, rho_diag_num[:, 1], label="Numerical solution", color="black", linewidth=2)
        axes[1].plot(t_test_interp, rho11_spline(t_test_interp), label="PINN solution", color="tab:green", linewidth=2)
        axes[1].set_xlabel("t (ps)")
        axes[1].set_ylabel(r"$\rho_{11}(t)$")
        axes[1].set_xlim(0,D)
        axes[1].set_ylim(0,1)
        axes[1].legend(fontsize=11)

        axes[2].plot(t_num, rho_diag_num[:, 2], label="Numerical solution", color="black", linewidth=2)
        axes[2].plot(t_test_interp, rho22_spline(t_test_interp), label="PINN solution", color="tab:blue", linewidth=2)
        axes[2].set_xlabel("t (ps)")
        axes[2].set_ylabel(r"$\rho_{22}(t)$")
        axes[2].set_xlim(0,D)
        axes[2].set_ylim(0,1)
        axes[2].legend(fontsize=11)
        '''
        axes[3].plot(t_test[:,0], abs(torch.vmap(torch.trace)(rho_pred)).detach().numpy(), color="black", linewidth=2)
        axes[3].set_xlabel("Time (ps)")
        axes[3].set_ylabel(r"Tr$(\rho(t))$")
        axes[3].set_xlim(0,D)
        axes[3].set_ylim(0,1.1)
        axes[3].hlines(y=1, xmin=0, xmax=D, linestyle='dotted', color='black')
        '''

        plt.tight_layout()

        plt.show()


    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        axes[0].plot(t_num, rho_diag_num[:,0], label="Numerical solution", color="black", linewidth=2)
        axes[0].plot(t_test_interp, rho00_spline(t_test_interp), label="PINN solution", color="tab:red", linewidth=2)
        axes[0].set_xlabel("t (ps)")
        axes[0].set_ylabel(r"$\rho_{00}(t)$")
        axes[0].set_xlim(0,D)
        axes[0].set_ylim(0,1)
        axes[0].legend(fontsize=11)

        axes[1].plot(t_num, rho_diag_num[:, 1], label="Numerical solution", color="black", linewidth=2)
        axes[1].plot(t_test_interp, rho11_spline(t_test_interp), label="PINN solution", color="tab:blue", linewidth=2)
        axes[1].set_xlabel("t (ps)")
        axes[1].set_ylabel(r"$\rho_{11}(t)$")
        axes[1].set_xlim(0,D)
        axes[1].set_ylim(0,1)
        axes[1].legend(fontsize=11)

        plt.tight_layout()

        plt.show()


def plot_test(times, rho_diag, t_test, results):
    plt.rcParams['font.size'] = 12.5

    fig, axes = plt.subplots(1,2, figsize=(8,3))

    axes[0].plot(times, rho_diag[:,0], label="Numerical", color="black", linewidth=1)
    axes[0].scatter(t_test, results[:, 0], label="PINN inference", color="tab:red", marker='*', s=45)
    axes[0].set_xlabel("t (ps)")
    axes[0].set_ylabel(r"$\rho_{00} (t)$")
    axes[0].legend(loc='lower left')

    axes[1].plot(times, rho_diag[:,1], label="Numerical", color="black", linewidth=1)
    axes[1].scatter(t_test, 1-results[:, 0], label="PINN inference", color="tab:blue", marker='*', s=45)
    axes[1].set_xlabel("t (ps)")
    axes[1].set_ylabel(r"$\rho_{11} (t)$")

    plt.tight_layout()
    plt.show()


def plot_midtrain_inv_1(t_obs, rho_obs, rho_obs_pred, alpha0, alpha_list):

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    axes[0].scatter(t_obs, torch.real(rho_obs[:,0,0]), label="Noisy data", alpha=0.6, linewidth=2, color="tab:red")
    axes[0].plot(t_obs, rho_obs_pred[:, 0], label="PINN solution", color="tab:red", linewidth=2)
    axes[0].set_xlabel("t (ps)")
    axes[0].set_ylabel(r"$\rho_{00} (t)$")
    axes[0].legend()

    axes[1].scatter(t_obs, torch.real(rho_obs[:,0,1]), label="Noisy data", alpha=0.6, linewidth=2, color="tab:blue")
    axes[1].plot(t_obs, rho_obs_pred[:, 1], label="PINN solution", color="tab:blue", linewidth=2)
    axes[1].set_xlabel("t (ps)")
    axes[1].set_ylabel(r"Re $[\rho_{01} (t)]$")
    axes[1].legend()
    
    axes[2].scatter(t_obs, torch.imag(rho_obs[:,0,1]), label="Noisy data", alpha=0.6, linewidth=2, color="tab:green")
    axes[2].plot(t_obs, rho_obs_pred[:, 2], label="PINN solution", color="tab:green", linewidth=2)
    axes[2].set_xlabel("t (ps)")
    axes[2].set_ylabel(r"Im $[\rho_{01} (t)]$")
    axes[2].legend()

    axes[3].plot(alpha_list, label="PINN estimate", color="darkorange", linewidth=2)
    axes[3].hlines(alpha0, 0, len(alpha_list), label=r"True $\alpha$", color="black", linewidth=2)
    axes[3].set_ylabel(r"$\alpha$")
    axes[3].set_xlabel("Training step")
    axes[3].legend()

    plt.tight_layout()
    plt.show()


def plot_midtrain_inv_2(t_obs, rho_obs, rho_obs_pred, v_c0, v_c_list):

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    axes[0].scatter(t_obs, torch.real(rho_obs[:,0,0]), label="Noisy data", alpha=0.6, linewidth=2, color="tab:red")
    axes[0].plot(t_obs, rho_obs_pred[:, 0], label="PINN solution", color="tab:red", linewidth=2)
    axes[0].set_xlabel("t (ps)")
    axes[0].set_ylabel(r"$\rho_{00} (t)$")
    axes[0].legend()

    axes[1].scatter(t_obs, torch.real(rho_obs[:,0,1]), label="Noisy data", alpha=0.6, linewidth=2, color="tab:blue")
    axes[1].plot(t_obs, rho_obs_pred[:, 1], label="PINN solution", color="tab:blue", linewidth=2)
    axes[1].set_xlabel("t (ps)")
    axes[1].set_ylabel(r"Re $[\rho_{01} (t)]$")
    axes[1].legend()
    
    axes[2].scatter(t_obs, torch.imag(rho_obs[:,0,1]), label="Noisy data", alpha=0.6, linewidth=2, color="tab:green")
    axes[2].plot(t_obs, rho_obs_pred[:, 2], label="PINN solution", color="tab:green", linewidth=2)
    axes[2].set_xlabel("t (ps)")
    axes[2].set_ylabel(r"Im $[\rho_{01} (t)]$")
    axes[2].legend()


    axes[3].plot(v_c_list, label="PINN estimate", color="darkorange", linewidth=2)
    axes[3].hlines(v_c0, 0, len(v_c_list), label=r"True $\omega_c$", color="black", linewidth=2)
    axes[3].set_ylabel(r"$\omega_c$")
    axes[3].set_xlabel("Training step")
    axes[3].legend()

    plt.tight_layout()
    plt.show()


def plot_sensitivity(dc_dalpha, dc_domega_c):
    plt.rcParams['font.size'] = 15
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    cf1 = axs[0].imshow(dc_dalpha.detach().numpy(), extent=[0, 5, 0, 5], origin='lower', aspect='auto', cmap='plasma')
    axs[0].set_title(r"(a)  $∂c(t, t')/∂α$")
    axs[0].set_xlabel("t'")
    axs[0].set_ylabel("t")
    fig.colorbar(cf1, ax=axs[0])

    cf2 = axs[1].imshow(dc_domega_c.detach().numpy(), extent=[0, 5, 0, 5], origin='lower', aspect='auto', cmap='plasma')
    axs[1].set_title(r"(b)  $∂ c(t, t')/∂ω_c$")
    axs[1].set_xlabel("t'")
    axs[1].set_ylabel("t")
    fig.colorbar(cf2, ax=axs[1])

    plt.tight_layout()
    plt.show()