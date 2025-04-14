"""
custom PyTorch Dataset class for loading and managing graph data
"""
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# Dataset class for loading graph data.
# Args:
# dInfo (dict):              Information about the dataset.
# dset_dir (str):            Directory where the dataset is located.
# length (int, optional):    Length of the dataset to load. Defaults to 0.
class GraphDataset(Dataset):
    def __init__(self, dInfo, dset_dir, length=0):
        self.dset_dir = dset_dir
        self.z_dim = len(dInfo['dataset']['state_variables'])
        self.q_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.dims = {
            'z': self.z_dim,
            'q': dInfo['dataset']['q_dim'],
            'q_0': dInfo['dataset']['q0_dim'],
            'n': 1,
            'f': dInfo['dataset']['external_force_dim'],
            'g':dInfo['dataset']['g_dim'],
        }

        self.dt = dInfo['dataset']['dt']
        self.data = torch.load(dset_dir, weights_only=False)
        if length != 0:
            self.data = self.data[:length]

    # Get data with index
    def __getitem__(self, index):
        data = self.data[index]
        return data

    # Get the length of the dataset
    def __len__(self):
        return len(self.data)

    # Compute statistics of the dataset
    def get_stats(self):
        total_tensor = torch.cat([data.x for data in self.data], dim=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(total_tensor)

        if self.dims['f'] == 0:
            scaler_f = None
        else:
            total_tensor_f = torch.cat([data.f for data in self.data], dim=0)
            scaler_f = MinMaxScaler(feature_range=(-1, 1))
            scaler_f.fit(total_tensor_f)
            scaler_f.min_[0] = 0

        return scaler, scaler_f


if __name__ == '__main__':
    pass
