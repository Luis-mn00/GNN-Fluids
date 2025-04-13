import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data, DataLoader
import wandb
from torch.utils.data import random_split
import os
import matplotlib.pyplot as plt
import json

from src.gnn_init import NodalGNN
from src.utils.utils import *

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

wandb.init(project="gnn-fluid-init")

def train(noise_var):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        # Add noise to the input features
        batch.x += torch.randn_like(batch.x) * noise_var
        optimizer.zero_grad()
        pred = model.pass_through_net(batch.x, batch.edge_index)
        loss = compute_loss_simple(pred, batch.y, batch.mask)  # Apply mask
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model.pass_through_net(batch.x, batch.edge_index)
            loss = compute_loss_simple(pred, batch.y, batch.mask)  # Apply mask
            total_loss += loss.item()
    return total_loss / len(val_loader)

def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model.pass_through_net(batch.x, batch.edge_index)
            loss = compute_loss_simple(pred, batch.y, batch.mask)  # Apply mask
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Load hyperparameters from JSON file
json_file_path = 'data_init/jsonFiles/dataset_Water3D.json'
with open(json_file_path, 'r') as f:
    hyperparams = json.load(f)

# Extract dataset and model parameters
dataset_params = hyperparams['dataset']
model_params = hyperparams['model']

# Use model parameters
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

# Update dataset paths
file_path_train = os.path.join(dataset_params['folder'], dataset_params['datasetPaths']['train'])
file_path_val = os.path.join(dataset_params['folder'], dataset_params['datasetPaths']['val'])
file_path_test = os.path.join(dataset_params['folder'], dataset_params['datasetPaths']['test'])

dataset_train = load_pt_dataset(file_path_train)
dataset_val = load_pt_dataset(file_path_val)
dataset_test = load_pt_dataset(file_path_test)

# Print the number of data points in each dataset
print(f"Number of training samples: {len(dataset_train)}")
print(f"Number of validation samples: {len(dataset_val)}")
print(f"Number of test samples: {len(dataset_test)}")

# Find max number of nodes for padding
max_nodes = max(max(data.y.shape[0] for data in dataset_train), 
                max(data.y.shape[0] for data in dataset_val), 
                max(data.y.shape[0] for data in dataset_test))

mean_x, std_x, mean_y, std_y, mean_vel, std_vel = compute_mean_std(dataset_train)

graphs_train = prepare_data(dataset_train, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes)
graphs_val = prepare_data(dataset_val, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes)
graphs_test = prepare_data(dataset_test, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes)

# Create DataLoader for each dataset
train_loader = DataLoader(graphs_train, batch_size=model_params["batch_size"], shuffle=True)
val_loader = DataLoader(graphs_val, batch_size=model_params["batch_size"], shuffle=False)
test_loader = DataLoader(graphs_test, batch_size=model_params["batch_size"], shuffle=False)

# === Step 6: Train, Validate, and Test the Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update model initialization
model = NodalGNN(n_hidden=n_hidden, dim_hidden=dim_hidden, num_passes=passes, input_dim=4).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=gamma)

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

# Crear la carpeta de checkpoints si no existe
name = f"train_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
save_folder = f'outputs_init/runs/{name}'
os.makedirs(save_folder, exist_ok=True)

# Initialize a variable to track the best validation loss
best_val_loss = float('inf')

for epoch in range(max_epoch):
    train_loss = train(noise_var)
    val_loss = validate()

    # Append losses to the lists
    if (epoch + 1) % 10 == 0:
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Check if the current validation loss is the best so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(save_folder, f"best_{epoch}_{val_loss}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"New best model saved with validation loss: {best_val_loss:.6f}")

    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": optimizer.param_groups[0]["lr"]
    })

    lr_scheduler.step()

    print(f"Epoch {epoch+1}/{max_epoch}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

# After training, plot the training history
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs', fontsize=14)  # Increase font size for axis title
plt.ylabel('Loss', fontsize=14)  # Increase font size for axis title
plt.yscale('log')  # Set y-axis to log scale
plt.ylim(0.1, 5)  # Set y-axis range between 0.1 and 5
plt.title('Training and Validation Loss History', fontsize=16)  # Increase font size for title
plt.legend()
plt.grid(True)
plt.xticks(fontsize=12)  # Increase font size for ticks
plt.yticks(fontsize=12)  # Increase font size for ticks

# Save the plot as an image
plot_path = os.path.join(save_folder, "training_history.png")
plt.savefig(plot_path)
plt.close()

# Test the model after training
test_loss = test()
print(f"Test Loss: {test_loss:.6f}")

wandb.finish()