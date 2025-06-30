import torch
import matplotlib.pyplot as plt


def plot_numerical(times_cpu, rho_diag_cpu, total_cpu, D):
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 15
    plt.plot(times_cpu, rho_diag_cpu[:, 0], label=r'$\rho_{00}$', linewidth=2)
    plt.plot(times_cpu, rho_diag_cpu[:, 1], label=r'$\rho_{11}$', linewidth=2)
    plt.plot(times_cpu, rho_diag_cpu[:, 2], label=r'$\rho_{22}$', linewidth=2)
    plt.plot(times_cpu, total_cpu, '--', label=r'Tr($\rho$)', color='black', linewidth=1.5)
    plt.xlabel('t (ps)')
    plt.ylabel('Population')
    plt.title('Numerical solution of NZ equation')
    plt.xlim(0, D)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_midtraining(t_num, rho_diag_num, t_test, rho_pred, D):

    rho_pred = rho_pred.detach()
    
    plt.rcParams['font.size'] = 12.5
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    axes[0].plot(t_num, rho_diag_num[:,0], label="Numerical solution", color="black", linewidth=2)
    axes[0].plot(t_test[:,0], torch.abs(rho_pred[:,0,0]), label="PINN solution", color="tab:red", linewidth=2)
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel(r"$|c_0(t)|^2$")
    axes[0].set_xlim(0,D)
    axes[0].set_ylim(0,1)
    axes[0].legend(fontsize=10)

    
    axes[1].plot(t_num, rho_diag_num[:, 1], label="Numerical solution", color="black", linewidth=2)
    axes[1].plot(t_test[:,0], torch.abs(rho_pred[:,1,1]), label="PINN solution", color="tab:green", linewidth=2)
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel(r"$|c_1(t)|^2$")
    axes[1].set_xlim(0,D)
    axes[1].set_ylim(0,1)
    axes[1].legend(fontsize=10)

    axes[2].plot(t_num, rho_diag_num[:, 2], label="Numerical solution", color="black", linewidth=2)
    axes[2].plot(t_test[:,0], torch.abs(rho_pred[:,2,2]), label="PINN solution", color="tab:blue", linewidth=2)
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylabel(r"$|c_2(t)|^2$")
    axes[2].set_xlim(0,D)
    axes[2].set_ylim(0,1)
    axes[2].legend(fontsize=10)

    axes[3].plot(t_test[:,0], abs(torch.vmap(torch.trace)(rho_pred)).detach().numpy(), color="black", linewidth=2)
    axes[3].set_xlabel("Time (ps)")
    axes[3].set_ylabel(r"Tr$(\rho(t))$")
    axes[3].set_xlim(0,D)
    axes[3].set_ylim(0,1.1)
    axes[3].hlines(y=1, xmin=0, xmax=D, linestyle='dotted', color='black')

    plt.tight_layout()

    plt.show()