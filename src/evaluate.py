import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from amb.metrics import rrmse_inf
from torch_geometric.loader import DataLoader
from src.utils.utils import print_error, generate_folder
from src.utils.plots import plot_2D_image, plot_2D, plot_image3D, plotError, plot_3D, video_plot_3D, plot_3D_mp
from src.utils.utils import compute_connectivity
from src.dataLoader.dataset import GraphDataset

from src.gnn_init import NodalGNN

def do_normalization(data, mean_x, std_x):
    data = (data - mean_x) / std_x
    return data

def undo_normalization(pred, mean_y, std_y):
    return (pred * std_y) + mean_y

def compute_error(z_net, z_gt, state_variables):
    # Compute error
    e = z_net.numpy() - z_gt.numpy()
    gt = z_gt.numpy()

    error = {clave: [] for clave in state_variables}
    L2_list = {clave: [] for clave in state_variables}

    epsilon = 1e-8  # Small value to avoid division by zero

    for i, sv in enumerate(state_variables):
        denominator = (gt[1:, :, i] ** 2).sum(1) + epsilon
        L2 = ((e[1:, :, i] ** 2).sum(1) / denominator) ** 0.5
        L22 = np.mean(((e[1:, :, i] ** 2).sum(1) / denominator)) ** 0.5
        error[sv] = L22
        L2_list[sv].extend(L2)

    return error, L2_list


def roll_out(nodal_gnn, start_gnn, dataloader, device, radius_connectivity, dtset_type, glass_flag=False):
    data = [sample for sample in dataloader]
    print(len(data))
    cnt_conet = 0
    cnt_gnn = 0

    # Cut the simulation to ignore the last 20 steps
    data = data[:-20]

    dim_z = data[0].x.shape[1]
    N_nodes = data[0].x.shape[0]
    if glass_flag:
        n = torch.zeros(len(data) + 1, N_nodes)
        n[0] = data[0].n
        N_nodes = data[0].x[n[0] == 1, :].shape[0]
    z_net = torch.zeros(len(data) + 1, N_nodes, dim_z)
    z_gt = torch.zeros(len(data) + 1, N_nodes, dim_z)

    # Initialize z_net_init to store the initial condition computed differently
    z_net_init = torch.zeros(len(data) + 1, N_nodes, dim_z)

    if glass_flag:
        z_net[0] = data[0].x[n[0] == 1]
        z_gt[0] = data[0].x[n[0] == 1]
    else:
        z_net[0] = data[0].x
        z_gt[0] = data[0].x

    # Load the .pth model from the given path
    start_gnn_model = NodalGNN(n_hidden=2, dim_hidden=64, num_passes=8, input_dim=4).to(device)
    start_gnn_model.load_state_dict(torch.load(start_gnn, map_location=device))
    start_gnn_model.eval()  # Set the model to evaluation mode

    # Modify the initial condition to maintain the first 3 components
    mean_x = torch.tensor([0.0543, 0.0252, 0.0550])
    std_x = torch.tensor([0.0294, 0.0240, 0.0293])
    mean_y = torch.tensor([1.6498e-02, -1.2449e-02, -9.6641e-03, 5.4860e-06])
    std_y = torch.tensor([2.3646e-01, 4.0701e-02, 1.8247e-01, 7.0457e-06])
    mean_vel = torch.tensor([0.0197])
    std_vel = torch.tensor([0.2772])
    
    #concatenate mean_x and std_x with mean_vel and std_vel
    mean_input = torch.cat((mean_x, mean_vel))
    std_input = torch.cat((std_x, std_vel))
        
    with torch.no_grad():
        first_3_components = data[0].x[:, :3]
        model_input = torch.cat((first_3_components, data[0].vel[:, 0].unsqueeze(1)), dim=1)
        model_input = do_normalization(model_input, mean_input, std_input)

        last_4_components = start_gnn_model.pass_through_net(model_input.to(device), data[0].edge_index)
        
        last_4_components = undo_normalization(last_4_components, mean_y, std_y)
        z_net_init[0] = torch.cat((first_3_components, last_4_components), dim=1)

    z_denorm = data[0].x
    edge_index = data[0].edge_index

    try:
        for t, snap in enumerate(data):
            snap.x = z_denorm
            snap.edge_index = edge_index
            snap = snap.to(device)
            with torch.no_grad():
                start_time = time.time()
                z_denorm, z_t1, _ = nodal_gnn.predict_step(snap, 1)
                cnt_gnn += time.time() - start_time
            if dtset_type == 'fluid':
                pos = z_denorm[:, :3].clone()
                start_time = time.time()
                edge_index = compute_connectivity(np.asarray(pos.cpu()), radius_connectivity, add_self_edges=False).to(
                    device)
                cnt_conet += time.time() - start_time
            else:
                edge_index = snap.edge_index
            if glass_flag:
                z_net[t + 1] = z_denorm[snap.n == 1]
                z_gt[t + 1] = z_t1[snap.n == 1]
            else:
                z_net[t + 1] = z_denorm
                z_gt[t + 1] = z_t1

        # Evolution for z_net_init
        z_denorm_init = z_net_init[0].clone()
        edge_index_init = data[0].edge_index
        for t, snap in enumerate(data):
            snap.x = z_denorm_init
            snap.edge_index = edge_index_init
            snap = snap.to(device)
            with torch.no_grad():
                z_denorm_init, _, _ = nodal_gnn.predict_step(snap, 1)
            if dtset_type == 'fluid':
                pos = z_denorm_init[:, :3].clone()
                edge_index_init = compute_connectivity(np.asarray(pos.cpu()), radius_connectivity, add_self_edges=False).to(
                    device)
            else:
                edge_index_init = snap.edge_index
            if glass_flag:
                z_net_init[t + 1] = z_denorm_init[snap.n == 1]
            else:
                z_net_init[t + 1] = z_denorm_init
    except:
        print(f'Ha fallado el rollout en el momento: {t}')

    print(f'El tiempo tardado en el compute connectivity: {cnt_conet}')
    print(f'El tiempo tardado en la red: {cnt_gnn}')
    return z_net, z_gt, z_net_init, t+1


def generate_results(plasticity_gnn, start_gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):
    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    save_dir_gif_pdc = os.path.join(output_dir_exp, f'result_pdc.gif')

    # Make roll out
    start_time = time.time()
    z_net, z_gt, z_net_init, t = roll_out(plasticity_gnn, start_gnn, test_dataloader, device, dInfo['dataset']['radius_connectivity'],
                              dInfo['dataset']['type'])
    print(f'El tiempo tardado en el rollout: {time.time() - start_time}')
    filePath = os.path.join(output_dir_exp, 'metrics.txt')
    with open(filePath, 'w') as f:
        error, L2_list = compute_error(z_net[1:, :, :], z_gt[1:, :, :], dInfo['dataset']['state_variables'])
        lines = print_error(error)
        error, L2_list = compute_error(z_net_init[1:, :, :], z_gt[1:, :, :], dInfo['dataset']['state_variables'])
        lines = print_error(error)
        f.write('\n'.join(lines))
        print("[Test Evaluation Finished]\n")
        f.close()
    plotError(z_gt, z_net, L2_list, dInfo['dataset']['state_variables'], dInfo['dataset']['dataset_dim'], output_dir_exp)

    if dInfo['project_name'] == 'Beam_2D':
        plot_2D_image(z_net, z_gt, -1, 4, output_dir=output_dir_exp)
        plot_2D(z_net, z_gt, save_dir_gif, var=4)
    else:
        video_plot_3D(z_net, z_net_init, z_gt, save_dir=save_dir_gif_pdc)
        plot_3D(z_net, z_net_init, z_gt, save_dir=save_dir_gif, var=-1)


