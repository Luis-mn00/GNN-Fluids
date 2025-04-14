"""
Initializer GNN model for the initial condition
"""
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch_scatter import scatter_mean

# Import necessary libraries for PyTorch, PyTorch Lightning, and graph operations

# Define a Multi-Layer Perceptron (MLP) class
# This class creates a feedforward neural network with customizable layers and activation functions
class MLP(torch.nn.Module):
    def __init__(self, layer_vec, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()  # Store layers in a list
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k + 1])  # Define linear layers
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)  # Apply linear transformation
            if i < len(self.layers) - 1:  # Apply activation function to all but the last layer
                x = nn.SiLU()(x)  # Use SiLU (Sigmoid-Weighted Linear Unit) activation
        return x

# Define a NodeModel class for node-level operations in the graph
# This class processes node features using edge attributes and updates node embeddings
class NodeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden):
        super(NodeModel, self).__init__()
        self.n_hidden = n_hidden  # Number of hidden layers
        self.dim_hidden = dim_hidden  # Dimension of hidden layers
        self.node_mlp = MLP([2 * dim_hidden] + n_hidden * [dim_hidden] + [dim_hidden])  # Define MLP for node updates

    def forward(self, x, dest, edge_attr):
        out = scatter_mean(edge_attr, dest, dim=0, dim_size=x.size(0))  # Aggregate edge attributes for each node
        out = torch.cat([x, out], dim=1)  # Concatenate node features with aggregated edge attributes
        out = self.node_mlp(out)  # Pass through the MLP
        return out

# Define an EdgeModel class for edge-level operations in the graph
# This class processes edge features using source and destination node features
class EdgeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden):
        super(EdgeModel, self).__init__()
        self.n_hidden = n_hidden  # Number of hidden layers
        self.dim_hidden = dim_hidden  # Dimension of hidden layers
        self.edge_mlp = MLP([3 * self.dim_hidden] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])  # Define MLP for edge updates

    def forward(self, src, dest, edge_attr):
        out = torch.cat([src, dest, edge_attr], dim=1)  # Concatenate source, destination, and edge features
        out = self.edge_mlp(out)  # Pass through the MLP
        return out

# Define a MetaLayer class for message passing in the graph
# This class combines edge and node models to perform graph updates
class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model  # Model for updating edge features
        self.node_model = node_model  # Model for updating node features

    def forward(self, x, edge_index, edge_attr, f=None):
        src, dest = edge_index  # Extract source and destination nodes from edge index
        edge_attr = self.edge_model(x[src], x[dest], edge_attr)  # Update edge features
        x = self.node_model(x, dest, edge_attr)  # Update node features
        return x, edge_attr

# Define the NodalGNN class for the graph neural network
# This class predicts velocities and energy using message passing and MLPs
class NodalGNN(pl.LightningModule):
    def __init__(self, n_hidden, dim_hidden, num_passes, input_dim, dropout_rate=0.5):
        super().__init__()
        self.num_passes = num_passes  # Number of message passing iterations

        # Encoder to project input features into a latent space
        self.encoder = MLP([input_dim] + n_hidden * [dim_hidden] + [dim_hidden], dropout_rate=dropout_rate)
        self.encoder_edge = MLP([1] + n_hidden * [dim_hidden] + [dim_hidden], dropout_rate=dropout_rate)

        # Define node and edge models for message passing
        self.node_model = NodeModel(n_hidden, dim_hidden)
        self.edge_model = EdgeModel(n_hidden, dim_hidden)

        # GraphNet combines node and edge models for message passing
        self.GraphNet = MetaLayer(node_model=self.node_model, edge_model=self.edge_model)

        # Decoder to project latent features back to output space (velocities and energy)
        self.decoder_vE = MLP([dim_hidden] + n_hidden * [dim_hidden] + [4], dropout_rate=dropout_rate)  # Output: vx, vy, vz, E

    def decoder(self, x):
        return self.decoder_vE(x)  # Decode latent features into output predictions

    def pass_through_net(self, pos, edge_index):
        # Forward pass through the GNN to predict velocities and energy

        # Step 1: Encode input positions into latent space
        x = self.encoder(pos)  
        
        src, dest = edge_index  # Extract source and destination nodes from edge index
        pos1 = x[src]  # Features of source nodes
        pos2 = x[dest]  # Features of destination nodes
        edge_attr = torch.norm(pos1 - pos2, dim=1, p=2).unsqueeze(1)  # Compute edge attributes (distance)
        edge_attr = self.encoder_edge(edge_attr)  # Encode edge attributes

        # Step 2: Perform multiple message passing iterations
        for _ in range(self.num_passes):
            x, edge_attr = self.GraphNet(x, edge_index, edge_attr)  # Update node and edge features

        # Step 3: Decode the final node features into velocities and energy
        vE_pred = self.decoder(x)  

        return vE_pred  # Return predictions
