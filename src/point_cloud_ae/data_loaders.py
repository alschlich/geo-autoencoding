import torch
import torch.utils.data
import numpy as np
import torch_geometric.data
import torch_geometric.loader
import pickle
from pytorch3d.structures import Meshes
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

class CADGraphDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)
        self.graphs = [d['graph'] for d in data]
        self.verts_list = [d['mesh'][0] for d in data]
        self.faces_list = [d['mesh'][1] for d in data]
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        mesh = Meshes(verts=[self.verts_list[index]], faces=[self.faces_list[index]])
        graph = self.graphs[index]
        return graph, mesh
    
    @staticmethod
    def collate_fn(batch):
        graphs, meshes = zip(*batch)

        graph_batch = Batch.from_data_list(graphs)
        mesh_batch = Meshes(
            verts=[m.verts_list()[0] for m in meshes],
            faces=[m.faces_list()[0] for m in meshes]
        )

        return graph_batch, mesh_batch

class CADPointcloudDataset(torch.utils.data.Dataset):
    def __init__(self,  path):
        dataset_np = np.load(path)
        self.data = torch.from_numpy(dataset_np).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_model(modelPath, modelClass):
    return modelClass.load_state_dict(torch.load(modelPath))

def split_dataset(dataset, train_set_percentage):
    lengths = [int(len(dataset)*train_set_percentage), len(dataset) - int(len(dataset) * train_set_percentage)]
    return torch.utils.data.random_split(dataset, lengths)

def get_pointcloud_dataloaders(path, batch_size, train_set_percentage, shuffle=True, num_workers=0, pin_memory=True):
    dataset = CADPointcloudDataset(path)
    train_set, test_set = split_dataset(dataset, train_set_percentage)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    return train_loader, test_loader

def get_graph_dataloaders(path, batch_size, train_set_percentage, shuffle=True):
    dataset = CADGraphDataset(path)
    train_set, test_set = split_dataset(dataset, train_set_percentage)
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=dataset.collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return train_loader, test_loader