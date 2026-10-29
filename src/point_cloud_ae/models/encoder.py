import torch
import pytorch3d.ops as ops
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse, remove_self_loops, to_undirected, degree
from torch.nn.utils.rnn import pad_sequence
from .vn_layers import *
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from geom_utils import batched_knn

class VNGEncoder(torch.nn.Module):
    def __init__(self, encoding_size):
        super().__init__()
        
        self.knn_feature_builder = VNGKNNFeatureBuilder()
        self.vndgcnn = VNDGCNN(encoding_size)

    def forward(self, x):
        x, valid_mask, position_for_vn, normal_for_vn = self.knn_feature_builder(x)
        x = self.vndgcnn(x, valid_mask, position_for_vn, normal_for_vn)
        return x

class VNDGCNN(torch.nn.Module):
    def __init__(self, encoding_size):
        super().__init__()
        
        self.encoding_size = encoding_size
        
        self.conv1 = VNLinearLeakyReLU(3, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)
        #self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2+1, 1, dim=4, share_nonlinearity=True)
        
        self.pool1 = VNMaskedMaxPool(64//3)
        self.pool2 = VNMaxPool(64//3)
        self.pool3 = VNMaxPool(128//3)
        self.pool4 = VNMaxPool(256//3)
        #self.pool5 = VNMaxPool(encoding_size)
        
        self.invar1 = VN1DInvariant(1)
    
    def forward(self, x, valid_mask, position, normal):
        # Shape localized feature composition/convolution
        x = self.conv1(x)
        x1 = self.pool1(x, valid_mask)
        # Feature space localized feature convolution
        x = get_graph_feature(x1)
        x = self.conv2(x)
        x2 = self.pool2(x)
        x = get_graph_feature(x2)
        x = self.conv3(x)
        x3 = self.pool3(x)
        x = get_graph_feature(x3)
        x = self.conv4(x)
        x4 = self.pool4(x)
        # Usually, compose global feature by concatenation and pooling
        # x5 = torch.cat((x1, x2, x3, x4), dim=1)
        #x = self.conv5(x5)
        #x = self.pool5(x)
        #Instead, concatenate features at different scales
        x6 = torch.cat((normal, x1, x2, x3, x4), dim=1)
        x7 = torch.cat((position, x6), dim = 1)
        # Idea: Assign attention, treat each point as a capsule
        #x = self.conv5(x6)
        #x = self.invar1(x).squeeze(1)
        #x = torch.softmax(x, dim = 1)
        # Choose a set number of capsules at runtime
        #values, idx = torch.topk(x, k = self.encoding_size)
        # Alternative for variable encoding size: Farthest point sampling weighted with attention based on attention treshold
        #idx = idx.unsqueeze(1).unsqueeze(1).expand(x7.size(0), x7.size(1), x7.size(2), -1)
        # Idea: Farthest point sampling to encoding size
        result = ops.sample_farthest_points(points = position.squeeze(1).transpose(-1, -2), K = self.encoding_size, random_start_point = True)
        idx = result[1]
        idx = idx.unsqueeze(1).unsqueeze(1).expand(x7.size(0), x7.size(1), x7.size(2), -1)
        # Collect capsules
        x = torch.gather(x7, -1, idx)
        return x

class VNGKNNFeatureBuilder(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")
    
    def forward(self, graph_batch):
        # Get one-hop neighbors for each node
        src, dst = graph_batch.edge_index
        sorted_src, perm = src.sort()
        sorted_dst = dst[perm]
        deg = degree(src, dtype=torch.long)
        neighbors_list = torch.split(sorted_dst, deg.tolist())
        neighbors_padded_mask = pad_sequence(neighbors_list, batch_first=True, padding_value=-1)
        neighbors_padded = pad_sequence(neighbors_list, batch_first=True, padding_value=0)
        valid_mask = neighbors_padded_mask > -1
        # Collect neighbor features
        neighbors_padded_expanded = neighbors_padded.unsqueeze(-1).expand(-1, -1, 3)
        neighbor_position_expanded = graph_batch.position.unsqueeze(1).expand(-1, neighbors_padded.size(-1), -1)
        neighbor_normal_expanded = graph_batch.normal.unsqueeze(1).expand(-1, neighbors_padded.size(-1), -1)
        neighbor_position_gathered = torch.gather(neighbor_position_expanded, 0, neighbors_padded_expanded)
        neighbor_normal_gathered = torch.gather(neighbor_normal_expanded, 0, neighbors_padded_expanded)
        # Calculate position difference from source node
        neighbor_position_diff = neighbor_position_gathered - graph_batch.position.unsqueeze(1).expand(neighbor_position_expanded.size())
        # Build feature: relative position, neighbor normal, source normal
        feature = torch.stack((neighbor_position_diff, neighbor_normal_gathered, graph_batch.normal.unsqueeze(1).expand(neighbor_position_diff.size())), dim = 1)
        # Convert to tensors for VNDGCNN
        # We rely on the fact that all graphs have the same node count
        feature_for_vn = feature.view(graph_batch.ptr.size(0) - 1, graph_batch.ptr[1], feature.size(1), feature.size(2), feature.size(3)).permute(0, 2, 4, 1, 3)
        valid_mask_for_vn = valid_mask.view(graph_batch.ptr.size(0) - 1, graph_batch.ptr[1], -1)
        position_for_vn = to_dense_batch(graph_batch.position, graph_batch.batch)[0].unsqueeze(1).transpose(-1, -2)
        normal_for_vn = to_dense_batch(graph_batch.normal, graph_batch.batch)[0].unsqueeze(1).transpose(-1, -2)
        return feature_for_vn, valid_mask_for_vn, position_for_vn, normal_for_vn