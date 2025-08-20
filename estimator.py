import torch
from torch import pi
from data.data_utils import *
from visualizations import plot_midtrain_inv_1, plot_midtrain_inv_2


def train_inv_1(t_obs, rho_obs, alpha0=0.126, v_c=3.04, D=1, T=10, t_intervals=20, M=50, 
                hidden_dim=32, num_layers=4, epochs=5000, lr=5e-3, lam=1e-2, weight_decay=1e-4,
                init_val=1, plot_interval=100, seed=2025):
    
    torch.manual_seed(seed)

    model = ExPINN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    t_physics = torch.linspace(0, D, t_intervals, device=device).view(-1, 1).requires_grad_(True)

    # treat cutoff frequency (v_c) as a learnable parameter, add it to optimiser
    alpha = torch.nn.Parameter(init_val*torch.ones(1, requires_grad=True))


    optimizer = torch.optim.Adam(list(model.parameters())+[alpha], lr=lr, weight_decay=weight_decay)


    # Sample system Hamiltonian
    om_b = pi / D
    Hs = om_b / 2 * sx
    #Hs = torch.tensor([[0.0, D], [D, 0.0]], dtype=torch.cfloat, device=device)

    #L_phy = []
    #L_data = []
    alpha_list = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(t_physics)
        _ , loss_pde = physics_loss(output=output, t_physics=t_physics, Hs=Hs, alpha=alpha, T=T, omega_c=v_c, sys='e')
        #print(rho00.shape)
        #mape0_list.append(mape(nz_short0[:-2], abs(rho[:-2,0,0].detach().numpy())))
        #mape1_list.append(mape(nz_short1[1:], abs(rho[1:,1,1].detach().numpy())))

        #data loss
        rho_obs_pred = model(t_obs)
        #print(rho_obs_pred)
        pred_dm = construct_dm(rho_obs_pred, sys='e')
        #print(pred_dm.shape)
        
        loss_data = torch.mean(torch.sum((torch.abs(rho_obs - pred_dm)**2).reshape(M,4), dim=1))
        
        loss = lam*loss_pde + loss_data
        alpha_list.append(alpha.item())
        #L_phy.append(loss_pde.item())
        #L_data.append(loss_data.item())

        loss.backward()
        optimizer.step()

        if epoch % plot_interval == 0:
            rho_obs_pred = rho_obs_pred.detach()
            print(f"Epoch {epoch}, Total Loss: {loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, Data Loss: {loss_data.item():.6f}, alpha = {alpha.item()}")
            plot_midtrain_inv_1(t_obs, rho_obs, rho_obs_pred, alpha0, alpha_list)

    return model



def train_inv_2(t_obs, rho_obs, v_c0=3.04, alpha=0.126, D=1, T=10, t_intervals=20, M=50, 
                hidden_dim=32, num_layers=4, epochs=5000, lr=5e-3, lam=1e-2, weight_decay=1e-4,
                init_val=4, plot_interval=100, seed=2025):
    
    torch.manual_seed(seed)

    model = ExPINN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    t_physics = torch.linspace(0, D, t_intervals, device=device).view(-1, 1).requires_grad_(True)

    # treat cutoff frequency (v_c) as a learnable parameter, add it to optimiser
    v_c = torch.nn.Parameter(init_val*torch.ones(1, requires_grad=True))


    optimizer = torch.optim.Adam(list(model.parameters())+[v_c], lr=lr, weight_decay=weight_decay)


    # Sample system Hamiltonian
    om_b = pi / D
    Hs = om_b / 2 * sx
    #Hs = torch.tensor([[0.0, D], [D, 0.0]], dtype=torch.cfloat, device=device)

    #L_phy = []
    #L_data = []
    v_c_list = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(t_physics)
        _ , loss_pde = physics_loss(output=output, t_physics=t_physics, Hs=Hs, alpha=alpha, T=T, omega_c=v_c, sys='e')
        #print(rho00.shape)
        #mape0_list.append(mape(nz_short0[:-2], abs(rho[:-2,0,0].detach().numpy())))
        #mape1_list.append(mape(nz_short1[1:], abs(rho[1:,1,1].detach().numpy())))
        #data loss
        rho_obs_pred = model(t_obs)
        #print(rho_obs_pred)
        pred_dm = construct_dm(rho_obs_pred, sys='e')
        #print(pred_dm)
        
        loss_data = torch.mean(torch.sum((torch.abs(rho_obs - pred_dm)**2).reshape(M,4), dim=1))
        
        loss = lam*loss_pde + loss_data

        v_c_list.append(v_c.item())
        #L_phy.append(loss_pde.item())
        #L_data.append(loss_data.item())

        loss.backward()
        optimizer.step()

        if epoch % plot_interval == 0:
            rho_obs_pred = rho_obs_pred.detach()
            print(f"Epoch {epoch}, Total Loss: {loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, Data Loss: {loss_data.item():.6f}, v_c = {v_c.item()}")
            plot_midtrain_inv_2(t_obs, rho_obs, rho_obs_pred, v_c0, v_c_list)

    return model