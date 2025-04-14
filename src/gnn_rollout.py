"""
Thermodynamics Informed GNN for the rollout
"""
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import add_self_loops

# Define a Multi-Layer Perceptron (MLP) class
# This class creates a feedforward neural network with customizable layers and activation functions
class MLP(torch.nn.Module):
    def __init__(self, layer_vec):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()  # Store layers in a list
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k + 1])  # Define linear layers
            self.layers.append(layer)
            self.layers.append(nn.SiLU()) if k != len(layer_vec) - 2 else None  # Add activation function except for the last layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Apply each layer sequentially
        return x

# Define an EdgeModel class for edge-level operations in the graph
# This class processes edge features using source and destination node features
class EdgeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden):
        super(EdgeModel, self).__init__()
        self.n_hidden = n_hidden  # Number of hidden layers
        self.dim_hidden = dim_hidden  # Dimension of hidden layers
        self.edge_mlp = MLP([3 * self.dim_hidden] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])  # Define MLP for edge updates

    def forward(self, src, dest, edge_attr):
        out = torch.cat([edge_attr, src, dest], dim=1)  # Concatenate edge attributes with source and destination features
        out = self.edge_mlp(out)  # Pass through the MLP
        return out

# Define a NodeModel class for node-level operations in the graph
# This class processes node features using edge attributes and updates node embeddings
class NodeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden, dims):
        super(NodeModel, self).__init__()
        self.n_hidden = n_hidden  # Number of hidden layers
        self.dim_hidden = dim_hidden  # Dimension of hidden layers
        if dims['f'] == 0:
            # Define MLP for node updates when no external force is present
            self.node_mlp = MLP([2 * self.dim_hidden + dims['f']] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])
        else:
            # Define MLP for node updates when external force is present
            self.node_mlp = MLP([2 * self.dim_hidden + int(1/2 * self.dim_hidden)] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, x, dest, edge_attr, f=None):
        out = scatter_mean(edge_attr, dest, dim=0, dim_size=x.size(0))  # Aggregate edge attributes for each node
        if f is not None:
            out = torch.cat([x, out, f], dim=1)  # Concatenate node features, aggregated edge attributes, and external force
        else:
            out = torch.cat([x, out], dim=1)  # Concatenate node features and aggregated edge attributes
        out = self.node_mlp(out)  # Pass through the MLP
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
        x = self.node_model(x, dest, edge_attr, f)  # Update node features
        return x, edge_attr

# Define the NodalGNN class for the graph neural network
# This class predicts system dynamics using message passing and MLPs
class NodalGNN(pl.LightningModule):
    def __init__(self, dims, scaler, dt_info, save_folder):
        super().__init__()
        n_hidden = dt_info['model']['n_hidden']  # Number of hidden layers
        dim_hidden = dt_info['model']['dim_hidden']  # Dimension of hidden layers
        self.project_name = dt_info['project_name']  # Name of the project
        self.passes = dt_info['model']['passes']  # Number of message passing iterations
        self.batch_size = dt_info['model']['batch_size']  # Batch size for training
        self.dtset_type = dt_info['dataset']['type']  # Type of dataset (e.g., fluid, solid)
        self.radius_connectivity = dt_info['dataset']['radius_connectivity']  # Radius for connectivity computation
        self.save_folder = save_folder  # Directory to save outputs
        self.data_dim = dt_info['dataset']['q_dim']  # Dimensionality of the data (2D or 3D)
        self.dims = dims  # Dimensions of various features
        self.dim_z = self.dims['z']  # Dimension of state variables
        self.dim_q = self.dims['q']  # Dimension of position variables
        dim_node = self.dims['z'] + self.dims['n'] - self.dims['q']  # Dimension of node features
        dim_edge = self.dims['q'] + self.dims['q_0'] + 1  # Dimension of edge features
        dim_f = self.dims['f']  # Dimension of external force features
        self.state_variables = dt_info['dataset']['state_variables']  # List of state variables

        # Define encoder MLPs for nodes and edges
        self.encoder_node = MLP([dim_node] + n_hidden * [dim_hidden] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden * [dim_hidden] + [dim_hidden])
        if self.dims['f'] > 0:
            self.encoder_f = MLP([dim_f] + n_hidden * [dim_hidden] + [int(dim_hidden / 2)])

        # Define node and edge models for message passing
        node_model = NodeModel(n_hidden, dim_hidden, self.dims)
        edge_model = EdgeModel(n_hidden, dim_hidden)
        self.GraphNet = MetaLayer(node_model=node_model, edge_model=edge_model)  # Combine node and edge models

        # Define decoders for gradients and other outputs
        self.decoder_E = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])  # Decoder for energy gradients
        self.decoder_S = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])  # Decoder for entropy gradients

        # Decoders for L and M matrices
        self.decoder_L = MLP([dim_hidden * 3] + n_hidden * [dim_hidden] * 2 + [int(self.dim_z * (self.dim_z + 1) / 2 - self.dim_z)])
        self.decoder_M = MLP([dim_hidden * 3] + n_hidden * [dim_hidden] * 2 + [int(self.dim_z * (self.dim_z + 1) / 2)])

        self.ones = torch.ones(self.dim_z, self.dim_z)  # Matrix of ones for constructing L and M matrices
        self.scaler, self.scaler_f = scaler  # Scalers for normalizing data
        self.dt = dt_info['dataset']['dt']  # Time step size
        self.noise_var = dt_info['model']['noise_var']  # Variance of noise added during training
        self.lambda_d = dt_info['model']['lambda_d']  # Weight for loss terms
        self.lr = dt_info['model']['lr']  # Learning rate
        self.miles = dt_info['model']['miles']  # Milestones for learning rate scheduler
        self.gamma = dt_info['model']['gamma']  # Decay factor for learning rate scheduler
        self.criterion = torch.nn.functional.mse_loss  # Loss function

        # Rollout simulation parameters
        self.rollout_freq = dt_info['model']['rollout_freq']  # Frequency of rollouts during training
        self.error_message_pass = []  # List to store error messages during message passing

    def decoder(self, x, edge_attr, src, dest):
        # Gradients
        dEdz = self.decoder_E(x).unsqueeze(-1)  # Compute energy gradients
        dSdz = self.decoder_S(x).unsqueeze(-1)  # Compute entropy gradients

        # Decode L and M matrices from edge attributes and node features
        l = self.decoder_L(torch.cat([edge_attr, x[src], x[dest]], dim=1))  # Decode lower triangular part of L
        m = self.decoder_M(torch.cat([edge_attr, x[src], x[dest]], dim=1))  # Decode lower triangular part of M

        # Initialize L and M matrices with zeros
        L = torch.zeros(edge_attr.size(0), self.dim_z, self.dim_z, device=l.device, dtype=l.dtype)
        M = torch.zeros(edge_attr.size(0), self.dim_z, self.dim_z, device=m.device, dtype=m.dtype)

        # Fill L and M matrices with decoded values
        L[:, torch.tril(self.ones, -1) == 1] = l
        M[:, torch.tril(self.ones) == 1] = m 

        # Compute symmetric versions of L and M
        Ledges = torch.subtract(L, torch.transpose(L, 1, 2))  
        Medges = torch.bmm(M, torch.transpose(M, 1, 2)) / (torch.max(M) + 1e-8)  

        # Separate diagonal and neighbor edges
        edges_diag = dest == src  
        edges_neigh = src != dest 

        # Compute contributions to dz/dt from L and M matrices
        L_dEdz = torch.matmul(Ledges, dEdz[dest, :, :])  
        M_dSdz = torch.matmul(Medges, dSdz[dest, :, :]) 

        # Compute total contribution for diagonal edges
        tot = (torch.matmul(Ledges[edges_diag, :, :], dEdz) + torch.matmul(Medges[edges_diag, :, :], dSdz))

        # Compute total contribution for neighbor edges
        M_dEdz_L_dSdz = L_dEdz + M_dSdz

        # Compute time derivative of z (dz/dt)
        dzdt_net = tot[:, :, 0] - scatter_add(M_dEdz_L_dSdz[:, :, 0][edges_neigh, :], src[edges_neigh], dim=0)

        # Compute degeneracy losses for energy and entropy
        loss_deg_E = (torch.matmul(Medges[edges_diag, :, :], dEdz)[:, :, 0] ** 2).mean()  # Energy degeneracy loss
        loss_deg_S = (torch.matmul(Ledges[edges_diag, :, :], dSdz)[:, :, 0] ** 2).mean()  # Entropy degeneracy loss

        return dzdt_net, loss_deg_E, loss_deg_S  

    def pass_thought_net(self, z_t0, z_t1, edge_index, n, f, batch=None, mode='val', plot_info = []):
        # Normalize input data (z_t0 and z_t1) using the scaler
        z_norm = torch.from_numpy(self.scaler.transform(z_t0.cpu())).float().to(self.device)
        z1_norm = torch.from_numpy(self.scaler.transform(z_t1.cpu())).float().to(self.device)

        # Normalize external force data if present
        if f is not None:
            f = torch.from_numpy(self.scaler_f.transform(f.cpu())).float().to(self.device)

        # Add noise to the normalized data during training for regularization
        if mode == 'train':
            noise = self.noise_var * torch.randn_like(z_norm[n == 0])
            z_norm[n == 0] = z_norm[n == 0] + noise * z_norm[n == 0]
            noise = self.noise_var * torch.randn_like(z_norm[n == 2])
            z_norm[n == 2] = z_norm[n == 2] + noise * z_norm[n == 2]

        # Split normalized data into position (q) and velocity (v) components
        q = z_norm[:, :self.dim_q]
        v = z_norm[:, self.dim_q:]
        x = torch.cat((v, torch.reshape(n.type(torch.float32), (len(n), 1))), dim=1)

        # Compute edge attributes based on position differences
        src, dest = edge_index
        u = q[src] - q[dest]  # Compute relative positions
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)  # Compute distances
        edge_attr = torch.cat((u, u_norm), dim=1)  # Combine relative positions and distances

        # Encode node and edge features
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)
        if f is not None:
            f = self.encoder_f(f)

        # Perform message passing through the graph network
        for i in range(self.passes):
            if mode == 'eval':
                plot_info.append(torch.norm(x, dim=1).reshape(-1, 1).clone())  # Store node feature norms for visualization
            x_res, edge_attr_res = self.GraphNet(x, edge_index, edge_attr, f=f)  # Update node and edge features
            x += x_res  # Residual connection for node features
            edge_attr += edge_attr_res  # Residual connection for edge features

        # Add self-loops to the graph for decoding
        if self.project_name == 'Glass3D':
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
            n_glass = x[n == 0].shape[0]  # Number of glass particles
            mask_fluid = ((edge_index >= n_glass)[0, :]) & ((edge_index >= n_glass)[1, :])  # Mask for fluid particles
            edge_index = edge_index[:, mask_fluid]  # Filter edges for fluid particles
            edge_index = edge_index - torch.min(edge_index)  # Reindex edges
            edge_attr = edge_attr[mask_fluid, :]  # Filter edge attributes
            x = x[n == 1]  # Filter node features for fluid particles
        else:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr)

        # Decode the time derivative of z (dz/dt) and degeneracy losses
        dzdt_net, loss_deg_E, loss_deg_S = self.decoder(x, edge_attr, edge_index[0, :], edge_index[1, :])

        # Compute the ground truth time derivative of z (dz/dt)
        dzdt = (z1_norm - z_norm) / self.dt

        # Handle special case for Glass3D project
        if self.project_name == 'Glass3D':
            dzdt_net_b = dzdt.clone()
            dzdt_net_b[n == 1] = dzdt_net  # Use predicted dz/dt for fluid particles
            dzdt = dzdt[n == 1]  # Filter ground truth dz/dt for fluid particles
        else:
            dzdt_net_b = dzdt_net.reshape(dzdt.shape)

        # Compute loss for the predicted dz/dt
        loss_z = self.criterion(dzdt, dzdt_net)

        # Combine losses (prediction loss and degeneracy losses)
        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)

        # Log losses during training and validation
        if mode != 'eval':
            self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            if mode == 'val':
                self.log(f"{mode}_deg_E", loss_deg_E, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_deg_S", loss_deg_S, prog_bar=False, on_step=False, on_epoch=True)

            # Log losses for individual state variables
            if self.state_variables is not None:
                for i, variable in enumerate(self.state_variables):
                    loss_variable = self.criterion(dzdt_net.reshape(dzdt.shape)[:, i], dzdt[:, i])
                    self.log(f"{mode}_loss_{variable}", loss_variable, prog_bar=True, on_step=False, on_epoch=True)

        torch.cuda.empty_cache()  # Clear GPU cache to free memory
        return dzdt_net_b, loss, plot_info  # Return predicted dz/dt, loss, and plot information

    def extrac_pass(self, batch, mode):
        # Extract data from the batch for processing
        if self.project_name == 'Beam_3D':
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n[:,0], batch.f
        elif self.project_name == 'Beam_2D':
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f
        else:
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, None

        # Pass the extracted data through the network
        dzdt_net, loss, plot_info = self.pass_thought_net(z_t0, z_t1, edge_index, n, f, batch=batch.batch, mode=mode)
        return dzdt_net, loss, plot_info

    def training_step(self, batch, batch_idx, g=None):
        # Perform a training step by passing the batch through the network
        dzdt_net, loss, _ = self.extrac_pass(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx, g=None):
        # Perform a validation step by passing the batch through the network
        self.extrac_pass(batch, 'val')

    def predict_step(self, batch, batch_idx, g=None):
        # Perform a prediction step to compute the next state
        dzdt_net, loss, plot_info = self.extrac_pass(batch, 'eval')
        z_norm = torch.from_numpy(self.scaler.transform(batch.x.cpu())).float().to(self.device)
        z1_net = z_norm + self.dt * dzdt_net  # Compute the next state using the predicted dz/dt
        z1_net_denorm = torch.from_numpy(self.scaler.inverse_transform(z1_net.detach().to('cpu'))).float().to(
            self.device)  # Denormalize the predicted state

        return z1_net_denorm, batch.y, plot_info  # Return the predicted state, ground truth, and plot information

    def configure_optimizers(self):
        # Configure the optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # Use Adam optimizer
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.miles, gamma=self.gamma),
            'monitor': 'train_loss'}  # Use MultiStepLR scheduler with milestones and decay factor

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}  # Return optimizer and scheduler
