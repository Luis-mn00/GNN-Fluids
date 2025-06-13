"""
Inference code to prediuct the initial condition
"""
import torch
import matplotlib.pyplot as plt
import os
from torch_geometric.data import DataLoader
import json

from src.gnn_init import NodalGNN
from src.utils.utils import *

# Import necessary libraries for inference and data handling
# This script performs inference to predict the initial conditions of a system

# Set environment variable to avoid library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load hyperparameters from a JSON file
json_file_path = 'data_init/jsonFiles/dataset_Water3D.json'
with open(json_file_path, 'r') as f:
    hyperparams = json.load(f)

# Extract dataset and model parameters from the JSON file
dataset_params = hyperparams['dataset']
model_params = hyperparams['model']

# Extract key model parameters for initialization
n_hidden = model_params['n_hidden']
dim_hidden = model_params['dim_hidden']
passes = model_params['passes']
lr = model_params['lr']
noise_var = model_params['noise_var']
batch_size = model_params['batch_size']
max_epoch = model_params['max_epoch']
miles = model_params['miles']
gamma = model_params['gamma']
weight_decay = model_params['weight_decay']

# Load training, validation, and test datasets
file_path = "data_init/dataset_oil_new/Glass_train.pt"  # Path to training dataset
dataset_train = load_pt_dataset(file_path)
print(f"Dataset size: {len(dataset_train)}")

file_path = "data_init/dataset_oil_new/Glass_val.pt"  # Path to validation dataset
dataset_val = load_pt_dataset(file_path)
print(f"Validation dataset size: {len(dataset_val)}")

file_path = "data_init/dataset_oil_new/Glass_test1.pt"  # Path to test dataset
dataset_test = load_pt_dataset(file_path)
#print(dataset_test)
print(f"Test dataset size: {len(dataset_test)}")

# Find max number of nodes for padding
max_nodes = max(max(data.x.shape[0] for data in dataset_train), 
                max(data.x.shape[0] for data in dataset_val), 
                max(data.x.shape[0] for data in dataset_test))

mean_x, std_x, mean_y, std_y, mean_vel, std_vel = compute_mean_std(dataset_train)
print(f"Mean x: {mean_x}, Std x: {std_x}")
print(f"Mean y: {mean_y}, Std y: {std_y}")
print(f"Mean vel: {mean_vel}, Std vel: {std_vel}")

graphs_train = prepare_data(dataset_train, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes)
graphs_val = prepare_data(dataset_val, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes)
graphs_test = prepare_data(dataset_test, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes)

# Create DataLoader for each dataset
train_loader = DataLoader(graphs_train, batch_size=model_params["batch_size"], shuffle=True)
val_loader = DataLoader(graphs_val, batch_size=model_params["batch_size"], shuffle=False)
test_loader = DataLoader(graphs_test, batch_size=model_params["batch_size"], shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update model initialization
model = NodalGNN(n_hidden=n_hidden, dim_hidden=dim_hidden, num_passes=passes, input_dim=4).to(device)
#model.load_state_dict(torch.load('data_init/weights/best_9247.pth', map_location=device))
#model.load_state_dict(torch.load('data_init/weights/best_1607.pth', map_location=device))
#model.load_state_dict(torch.load('data_init/weights/best_3071.pth', map_location=device))
#model.load_state_dict(torch.load('data_init/weights/best_1629.pth', map_location=device))
#model.load_state_dict(torch.load('data_init/weights/best_2356.pth', map_location=device))
model.load_state_dict(torch.load('data_init/weights/best_890.pth', map_location=device))

# Create folder to save the plots
output_folder = "outputs_init/plots"
os.makedirs(output_folder, exist_ok=True)

# Iterate over all validation glasses and plot
for idx, val_sample in enumerate(val_loader):
    val_sample = val_sample.to(device)
    pred = model.pass_through_net(val_sample.x, val_sample.edge_index)

    # Convert to numpy for plotting
    pos_x, pos_y, pos_z = val_sample.x[:, 0].cpu().numpy(), val_sample.x[:, 1].cpu().numpy(), val_sample.x[:, 2].cpu().numpy()
    vel_x, vel_y, vel_z = pred[:, 0].detach().cpu().numpy(), pred[:, 1].detach().cpu().numpy(), pred[:, 2].detach().cpu().numpy()
    vel_x_gt, vel_y_gt, vel_z_gt = val_sample.y[:, 0].cpu().numpy(), val_sample.y[:, 1].cpu().numpy(), val_sample.y[:, 2].cpu().numpy()

    # Create a 3x3 figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Plot Velocity X vs Position X
    axes[0, 0].scatter(pos_x, vel_x, s=1, color="blue", label="Predicted")
    axes[0, 0].scatter(pos_x, vel_x_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[0, 0].set_xlabel("Position X")
    axes[0, 0].set_ylabel("Velocity X")
    axes[0, 0].set_title("Velocity X vs Position X")
    axes[0, 0].legend()

    # Plot Velocity X vs Position Y
    axes[0, 1].scatter(pos_y, vel_x, s=1, color="blue", label="Predicted")
    axes[0, 1].scatter(pos_y, vel_x_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[0, 1].set_xlabel("Position Y")
    axes[0, 1].set_ylabel("Velocity X")
    axes[0, 1].set_title("Velocity X vs Position Y")
    axes[0, 1].legend()

    # Plot Velocity X vs Position Z
    axes[0, 2].scatter(pos_z, vel_x, s=1, color="blue", label="Predicted")
    axes[0, 2].scatter(pos_z, vel_x_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[0, 2].set_xlabel("Position Z")
    axes[0, 2].set_ylabel("Velocity X")
    axes[0, 2].set_title("Velocity X vs Position Z")
    axes[0, 2].legend()

    # Plot Velocity Y vs Position X
    axes[1, 0].scatter(pos_x, vel_y, s=1, color="blue", label="Predicted")
    axes[1, 0].scatter(pos_x, vel_y_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[1, 0].set_xlabel("Position X")
    axes[1, 0].set_ylabel("Velocity Y")
    axes[1, 0].set_title("Velocity Y vs Position X")
    axes[1, 0].legend()

    # Plot Velocity Y vs Position Y
    axes[1, 1].scatter(pos_y, vel_y, s=1, color="blue", label="Predicted")
    axes[1, 1].scatter(pos_y, vel_y_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[1, 1].set_xlabel("Position Y")
    axes[1, 1].set_ylabel("Velocity Y")
    axes[1, 1].set_title("Velocity Y vs Position Y")
    axes[1, 1].legend()

    # Plot Velocity Y vs Position Z
    axes[1, 2].scatter(pos_z, vel_y, s=1, color="blue", label="Predicted")
    axes[1, 2].scatter(pos_z, vel_y_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[1, 2].set_xlabel("Position Z")
    axes[1, 2].set_ylabel("Velocity Y")
    axes[1, 2].set_title("Velocity Y vs Position Z")
    axes[1, 2].legend()

    # Plot Velocity Z vs Position X
    axes[2, 0].scatter(pos_x, vel_z, s=1, color="blue", label="Predicted")
    axes[2, 0].scatter(pos_x, vel_z_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[2, 0].set_xlabel("Position X")
    axes[2, 0].set_ylabel("Velocity Z")
    axes[2, 0].set_title("Velocity Z vs Position X")
    axes[2, 0].legend()

    # Plot Velocity Z vs Position Y
    axes[2, 1].scatter(pos_y, vel_z, s=1, color="blue", label="Predicted")
    axes[2, 1].scatter(pos_y, vel_z_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[2, 1].set_xlabel("Position Y")
    axes[2, 1].set_ylabel("Velocity Z")
    axes[2, 1].set_title("Velocity Z vs Position Y")
    axes[2, 1].legend()

    # Plot Velocity Z vs Position Z
    axes[2, 2].scatter(pos_z, vel_z, s=1, color="blue", label="Predicted")
    axes[2, 2].scatter(pos_z, vel_z_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[2, 2].set_xlabel("Position Z")
    axes[2, 2].set_ylabel("Velocity Z")
    axes[2, 2].set_title("Velocity Z vs Position Z")
    axes[2, 2].legend()

    # Set the same axis limits for all subplots
    x_limits = (-2, 2)
    y_limits = (-4, 4)

    for ax in axes.flat:
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

    # Adjust font sizes for titles, axis labels, and ticks
    font_size_title = 16
    font_size_labels = 14
    font_size_ticks = 14

    for ax in axes.flat:
        ax.title.set_fontsize(font_size_title)
        ax.xaxis.label.set_fontsize(font_size_labels)
        ax.yaxis.label.set_fontsize(font_size_labels)
        ax.tick_params(axis='both', which='major', labelsize=font_size_ticks)

    plt.tight_layout()

    # Save image for each validation glass
    file_path = os.path.join(output_folder, f"velocities_val_{idx}.png")
    plt.savefig(file_path)
    plt.close(fig)