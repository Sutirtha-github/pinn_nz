import torch
from scipy.interpolate import make_interp_spline
from data.data_utils import *
from visualizations import plot_midtrain_sim


# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Training for exciton + biexciton systems
def train_nz_pinn(sys, t, rho_diag, D=0.1, t_intervals = 20, alpha=0.126, T=100, omega_c=3.04,
                  n_layers=4, n_hidden=32, epochs=5001, lr=1e-2, lam=0.001, plot_interval=500):
    
    '''
    Train the PINN to learn NZ EOM

    Args:
        sys: 'b' for bieciton (default) and 'e' for exciton model
        t: time points of numerical solution 
        rho_diag: population (diagonal elements of density matrix) of ground, exciton and biexciton levels at each t
        D: pulse duration (ps)
        t_intervals: # time points in [0,D] for training
        A: system-bath coupling strength (ps/K)
        T: bath temperature (K)
        omega_c: bath cutoff frequency (1/ps)
        n_layers: # hidden layers
        n_hidden: # units per hidden layer
        epochs: # training iterations
        lr: learning rate
        plot_interval: frequency of plotting learnt dynamics

    Returns:
        trained model
    '''

    torch.manual_seed(2025)

    t_boundary = torch.tensor([[0.0]]).requires_grad_(True)
    t_physics = torch.linspace(0, D, t_intervals, device=device).view(-1, 1).requires_grad_(True)
    t_test = torch.linspace(0, D, t_intervals, device=device).view(-1, 1)

    rho_diag_cpu = rho_diag.detach().cpu().numpy()
    t_cpu = t.detach().cpu().numpy()

    rho00_spline = make_interp_spline(t_cpu, rho_diag_cpu[:,0])
    rho11_spline = make_interp_spline(t_cpu, rho_diag_cpu[:,1])
    rho00_short  = rho00_spline(t_test)
    rho11_short  = rho11_spline(t_test)
    
    if sys == 'b':
        rho22_spline = make_interp_spline(t_cpu, rho_diag_cpu[:,2])
        rho22_short  = rho22_spline(t_test)
    
    

    om_b = torch.pi / D
    Hs = hamilton(om_b, sys)

    if sys == 'b' : model = BiexPINN(n_hidden, n_layers).to(device)
    else : model = ExPINN(n_hidden, n_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(t_physics)
        output0 = model(t_boundary)

        rho , loss_physics = physics_loss(output, t_physics, Hs, alpha, T, omega_c, sys)
        loss_boundary = boundary_loss(output0, sys)
        loss =   lam * loss_physics +  loss_boundary  

        loss.backward()
        optimizer.step()
       
        if (epoch+1) % plot_interval == 0:
            #rho = construct_dm(output, sys).detach()
            err = []
            err += [rmse(rho00_short, rho[:,0,0]), rmse(rho11_short, rho[:,1,1])]
            if sys=='b':
                err += [rmse(rho22_short, rho[:,2,2])]
            print(f"Epoch: {epoch}  |  Loss = {loss.item():.5e}  |  RMSE = {torch.stack(err)}")
            plot_midtrain_sim(t, rho_diag, t_test, rho, D, sys)

    return model

