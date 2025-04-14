"""
Training code for the initilization of the initial conditions
"""
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
import wandb
import os
import matplotlib.pyplot as plt
import json

from src.gnn_init import NodalGNN
from src.utils.utils import *

# Initialize Weights & Biases (WandB) for experiment tracking
wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")
wandb.init(project="gnn-fluid-init")

# Define the training function
# Trains the model for one epoch and returns the average training loss
def train(noise_var):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        batch.x += torch.randn_like(batch.x) * noise_var  # Add noise for regularization
        optimizer.zero_grad()
        pred = model.pass_through_net(batch.x, batch.edge_index)
        loss = compute_loss_simple(pred, batch.y, batch.mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Define the validation function
# Evaluates the model on the validation set and returns the average validation loss
def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model.pass_through_net(batch.x, batch.edge_index)
            loss = compute_loss_simple(pred, batch.y, batch.mask)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Define the test function
# Evaluates the model on the test set and returns the average test loss
def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model.pass_through_net(batch.x, batch.edge_index)
            loss = compute_loss_simple(pred, batch.y, batch.mask)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Load hyperparameters from a JSON file
json_file_path = 'data_init/jsonFiles/dataset_Water3D.json'
with open(json_file_path, 'r') as f:
    hyperparams = json.load(f)  # Load hyperparameters as a dictionary

# Extract dataset and model parameters from the JSON file
dataset_params = hyperparams['dataset']
model_params = hyperparams['model']

# Extract model hyperparameters
n_hidden = model_params['n_hidden']  # Number of hidden layers
dim_hidden = model_params['dim_hidden']  # Dimension of hidden layers
passes = model_params['passes']  # Number of message passing steps
lr = model_params['lr']  # Learning rate
noise_var = model_params['noise_var']  # Noise variance for input features
batch_size = model_params['batch_size']  # Batch size
max_epoch = model_params['max_epoch']  # Maximum number of epochs
miles = model_params['miles']  # Milestones for learning rate scheduler
gamma = model_params['gamma']  # Gamma for learning rate scheduler
weight_decay = model_params['weight_decay']  # Weight decay for optimizer

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

# Prepare datasets and compute statistics
mean_x, std_x, mean_y, std_y, mean_vel, std_vel = compute_mean_std(dataset_train)  # Compute mean and std for normalization

graphs_train = prepare_data(dataset_train, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes)
graphs_val = prepare_data(dataset_val, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes)
graphs_test = prepare_data(dataset_test, mean_x, std_x, mean_y, std_y, mean_vel, std_vel, max_nodes)

# Create DataLoader for each dataset
train_loader = DataLoader(graphs_train, batch_size=model_params["batch_size"], shuffle=True)  # Training DataLoader
val_loader = DataLoader(graphs_val, batch_size=model_params["batch_size"], shuffle=False)  # Validation DataLoader
test_loader = DataLoader(graphs_test, batch_size=model_params["batch_size"], shuffle=False)  # Test DataLoader

# === Step 6: Train, Validate, and Test the Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, optimizer, and learning rate scheduler
model = NodalGNN(n_hidden=n_hidden, dim_hidden=dim_hidden, num_passes=passes, input_dim=4).to(device)  # Initialize model
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # AdamW optimizer
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=gamma)  # Learning rate scheduler

# Initialize lists to store training and validation losses
train_losses = []  # List to store training losses
val_losses = []  # List to store validation losses

# Create a folder to save checkpoints
name = f"train_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"  # Generate a unique folder name
save_folder = f'outputs_init/runs/{name}'  # Define the save folder path
os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Initialize a variable to track the best validation loss
best_val_loss = float('inf')  # Set initial best validation loss to infinity

# Training loop
for epoch in range(max_epoch):
    train_loss = train(noise_var)  # Train the model for one epoch
    val_loss = validate()  # Validate the model

    # Append losses to the lists every 10 epochs
    if (epoch + 1) % 10 == 0:
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Check if the current validation loss is the best so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss  # Update the best validation loss
        checkpoint_path = os.path.join(save_folder, f"best_{epoch}_{val_loss}.pth")  # Define checkpoint path
        torch.save(model.state_dict(), checkpoint_path)  # Save the model checkpoint
        print(f"New best model saved with validation loss: {best_val_loss:.6f}")

    # Log metrics to WandB
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": optimizer.param_groups[0]["lr"]
    })

    lr_scheduler.step()  # Update the learning rate

    # Print epoch summary
    print(f"Epoch {epoch+1}/{max_epoch}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training and Validation Loss History')
plt.legend()
plt.grid(True)
plot_path = os.path.join(save_folder, "training_history.png")
plt.savefig(plot_path)
plt.close()

# Test the model after training
test_loss = test()
print(f"Test Loss: {test_loss:.6f}")

wandb.finish()