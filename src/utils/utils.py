"""utils.py"""

import os
import shutil
import numpy as np
import argparse
import torch
from sklearn import neighbors
import datetime
from torch_geometric.data import Data


def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_error(error):
    print('State Variable (L2 relative error)')
    lines = []

    for key in error.keys():
        e = error[key]
        # error_mean = sum(e) / len(e)
        # line = '  ' + key + ' = {:1.2e}'.format(error_mean)
        line = '  ' + key + ' = {:1.2e}'.format(e)
        print(line)
        lines.append(line)
    return lines


def compute_connectivity(positions, radius, add_self_edges):
    """Get the indices of connected edges with radius connectivity.
    https://github.com/deepmind/deepmind-research/blob/master/learning_to_simulate/connectivity_utils.py
    Args:
      positions: Positions of nodes in the graph. Shape:
        [num_nodes_in_graph, num_dims].
      radius: Radius of connectivity.
      add_self_edges: Whether to include self edges or not.
    Returns:
      senders indices [num_edges_in_graph]
      receiver indices [num_edges_in_graph]
    """
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    if not add_self_edges:
        # Remove self edges.
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]

    return torch.from_numpy(np.array([senders, receivers]))


def generate_folder(output_dir_exp, path_dinfo, path_weights):
    if os.path.exists(output_dir_exp):
        print("The experiment path exists.")
        action = input("¿Would you like to create a new one (c) or overwrite (o)?")
        if action == 'c':
            output_dir_exp = output_dir_exp + '_new'
            os.makedirs(output_dir_exp, exist_ok=True)
    else:
        os.makedirs(output_dir_exp, exist_ok=True)

    shutil.copyfile(os.path.join('data_rollout', 'jsonFiles', path_dinfo),
                    os.path.join(output_dir_exp, os.path.basename(path_dinfo)))
    shutil.copyfile(os.path.join('data_rollout', 'weights', path_weights),
                    os.path.join(output_dir_exp, os.path.basename(path_weights)))
    return output_dir_exp

def load_diff_weights(checkpoint, nodal_gnn):
    old_state_dict = checkpoint['state_dict']
    new_state_dict = nodal_gnn.state_dict()
    updated_state_dict = {k: v for k, v in old_state_dict.items() if
                          k in new_state_dict and v.shape == new_state_dict[k].shape}

    new_state_dict.update(updated_state_dict)
    return new_state_dict

def load_pt_dataset(file_path):
    dataset = torch.load(file_path, weights_only=False)
    return dataset

def compute_mean_std(dataset):
    all_x = torch.cat([data.y[:, :3] for data in dataset], dim=0)  # Positions
    all_y = torch.cat([data.y[:, 3:7] for data in dataset], dim=0)  # Velocities + Energy
    all_vel = torch.cat([data.vel for data in dataset], dim=0) 
    
    mean_x, std_x = all_x.mean(dim=0), all_x.std(dim=0)
    mean_y, std_y = all_y.mean(dim=0), all_y.std(dim=0)
    mean_vel, std_vel = all_vel.mean(dim=0), all_vel.std(dim=0)
    
    std_x[std_x == 0] = 1
    std_y[std_y == 0] = 1
    std_vel[std_vel == 0] = 1
    
    return mean_x, std_x, mean_y, std_y, mean_vel, std_vel

def prepare_data(dataset, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes):
    graphs = []
    for data in dataset:
        num_nodes = data.x.shape[0]

        # Normalizar datos
        input_data = (data.y[:, :3] - mean_x) / std_x
        vel_data = (data.vel[:, 0] - mean_vel[0]) / std_vel[0]
        # Concatenate vel_data and input_data
        input_data = torch.cat((input_data, vel_data.unsqueeze(1)), dim=1)
        
        output_data = (data.y[:, 3:7] - mean_y) / std_y  

        # Padding con ceros
        input_padded = torch.zeros((max_nodes, input_data.shape[1]))
        output_padded = torch.zeros((max_nodes, output_data.shape[1]))

        input_padded[:num_nodes] = input_data
        output_padded[:num_nodes] = output_data

        # Crear máscara para nodos válidos
        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:num_nodes] = 1  # 1 para nodos reales, 0 para padding

        # Crear objeto PyG Data
        graph = Data(x=input_padded, edge_index=data.edge_index, y=output_padded, mask=mask)
        graphs.append(graph)

    return graphs

# === Step 5: Define Custom Loss Function ===
def compute_loss(pred, target, mask):
    mask = mask.unsqueeze(-1)  # Expandir dimensiones para broadcasting
    
    # Calcular MSE individualmente para cada variable
    loss_vx = ((pred[:, 0] - target[:, 0]) ** 2 * mask).sum() / mask.sum()
    loss_vy = ((pred[:, 1] - target[:, 1]) ** 2 * mask).sum() / mask.sum()
    loss_vz = ((pred[:, 2] - target[:, 2]) ** 2 * mask).sum() / mask.sum()
    loss_energy = ((pred[:, 3] - target[:, 3]) ** 2 * mask).sum() / mask.sum()

    total_loss = loss_vx + loss_vy + loss_vz + loss_energy

    return total_loss

def compute_loss_simple(pred, target, mask):
    mask = mask.unsqueeze(-1)  # Expandir dimensiones para hacer broadcasting

    # Calcular MSE y aplicar máscara
    loss = (pred - target) ** 2
    loss = loss * mask  # Ignorar nodos de padding
    loss = loss.sum() / mask.sum()  # Promedio solo en nodos válidos

    return loss


