import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add, scatter_mean

# Multi Layer Perceptron (MLP) class without Dropout
class MLP(torch.nn.Module):
    def __init__(self, layer_vec, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k + 1])
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation except for the last layer
                x = nn.SiLU()(x)
        return x

    
class NodeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden):
        super(NodeModel, self).__init__()
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.node_mlp = MLP([2*dim_hidden] + n_hidden * [dim_hidden] + [dim_hidden])

    def forward(self, x, dest, edge_attr):
        out = scatter_mean(edge_attr, dest, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp(out)
        return out

    
class EdgeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden):
        super(EdgeModel, self).__init__()
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.edge_mlp = MLP([3 * self.dim_hidden] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, src, dest, edge_attr):
        out = torch.cat([src, dest, edge_attr], dim=1)
        out = self.edge_mlp(out)
        return out

class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, f=None):
        src, dest = edge_index
        edge_attr = self.edge_model(x[src], x[dest], edge_attr)
        x = self.node_model(x, dest, edge_attr)
        return x, edge_attr


# Final model (simplified GNN for velocity and energy prediction)
class NodalGNN(pl.LightningModule):
    def __init__(self, n_hidden, dim_hidden, num_passes, input_dim, dropout_rate=0.5):
        super().__init__()
        self.num_passes = num_passes

        # Encoder to project input positions to latent space
        self.encoder = MLP([input_dim] + n_hidden * [dim_hidden] + [dim_hidden], dropout_rate=dropout_rate)
        self.encoder_edge = MLP([1] + n_hidden * [dim_hidden] + [dim_hidden], dropout_rate=dropout_rate)

        # Node model and Edge model
        self.node_model = NodeModel(n_hidden, dim_hidden)
        self.edge_model = EdgeModel(n_hidden, dim_hidden)

        # GNN MessagePassing Layer
        self.GraphNet = MetaLayer(node_model=self.node_model, edge_model=self.edge_model)

        # Decoder to project latent features back to velocities and energy
        self.decoder_vE = MLP([dim_hidden] + n_hidden * [dim_hidden] + [4], dropout_rate=dropout_rate)  # Output 4 values: vx, vy, vz, E

    def decoder(self, x):
        return self.decoder_vE(x)

    def pass_through_net(self, pos, edge_index):
        """ Forward pass through the GNN to predict velocities and energy. """
        # Step 1: Encode input positions into latent space
        x = self.encoder(pos)  
        
        src, dest = edge_index
        pos1 = x[src] 
        pos2 = x[dest] 
        edge_attr = torch.norm(pos1 - pos2, dim=1, p=2).unsqueeze(1)
        edge_attr = self.encoder_edge(edge_attr)

        # Step 2: Multiple message passing passes
        for _ in range(self.num_passes):
            x, edge_attr = self.GraphNet(x, edge_index, edge_attr)

        # Step 3: Decode the final node features into velocities and energy
        vE_pred = self.decoder(x)  

        return vE_pred
