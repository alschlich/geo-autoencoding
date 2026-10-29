import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse, remove_self_loops
from .vn_layers import *
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from geom_utils import sample_in_circle_rotated, batched_knn

class VNGDecoder(nn.Module):
    def __init__(self, encoding_size, out_node_count):
        super().__init__()
        
        self.eps = 1e-10
        
        self.encoding_size = encoding_size
        self.out_node_count = out_node_count
        
        self.conv1 = VNLinearLeakyReLU(256//3+128//3+64//3*2+1, 256//3+128//3+64//3*2+1, dim = 4)
        self.conv2 = VNLinearLeakyReLU(256//3+128//3+64//3*2+1, 256//3+128//3+64//3*2+1, dim = 4)
        self.lin1 = VNLinear(256//3+128//3+64//3*2+1, 20)

    def forward(self, x):
        position = x[:, 0, :, :].unsqueeze(1)
        normal = x[:, 1, :, :].unsqueeze(1)
        # Decode locally per capsule
        x = x[:, 1:, :, :]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.lin1(x)
        # Reform relative position
        positition_predicted = x.unsqueeze(-1).transpose(1, -1)
        # Normalize to set length
        positition_predicted = torch.nn.functional.normalize(positition_predicted, dim = 2) * 0.05
        # Calculate pairwise distance loss
        positition_predicted_pairwise = positition_predicted.squeeze(1).permute(0, 2, 3, 1)
        diff = positition_predicted_pairwise.unsqueeze(3) - positition_predicted_pairwise.unsqueeze(2)
        pairwise_distance_loss = - 100 * (diff ** 2).sum() / positition_predicted.size(0) / positition_predicted.size(-2) / positition_predicted.size(-1)
        # Move to absolute position
        position_expanded = position.unsqueeze(-1).expand(-1, -1, -1, -1, positition_predicted.size(-1))
        positition_predicted = positition_predicted + position_expanded
        # Reshape to output dimensions
        x = positition_predicted.reshape(positition_predicted.size(0), 1, 3, -1)
        data_list = []
        for i, node_feat in enumerate(x.unbind(dim=0)):
            data_list.append(Data(edge_index=None, position=node_feat[0, :, :].permute(1, 0), num_nodes=node_feat.size(-1)))
        graph_batch_updated = Batch.from_data_list(data_list)
        return graph_batch_updated, pairwise_distance_loss