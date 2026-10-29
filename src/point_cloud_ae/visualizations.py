import sys
import os
import numpy as np
import time
import torch
import torch.optim as optim
import point_cloud_ae.models.encoder as encoder
import point_cloud_ae.models.decoder as decoder
import point_cloud_ae.data_loaders as data_loaders
import point_cloud_ae.vis_utils as vis_utils
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Set parameters
input_node_count = 2048
output_node_count = input_node_count
encoding_size = 1024
batch_size = 1
train_set_percentage = 0.9
data_input_path = "data/dataset_model_net_40.pkl"
model_input_path = "output/best_model.pth"

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Detect CUDA-device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Make encoder network
encoder_instance = encoder.VNGEncoder(encoding_size)
encoder_instance.to(device)

# Make decoder network
decoder_instance = decoder.VNGDecoder(encoding_size, input_node_count)
decoder_instance.to(device)

# Load data
train_data_loader, test_data_loader = data_loaders.get_graph_dataloaders(data_input_path, batch_size, train_set_percentage)

# Load model
checkpoint = torch.load(model_input_path, map_location=torch.device(device))
encoder_instance.load_state_dict(checkpoint["encoder"])
decoder_instance.load_state_dict(checkpoint["decoder"])

# Print model stats
print("Encoder parameter count:")
print(sum(p.numel() for p in encoder_instance.parameters() if p.requires_grad))

print("Decoder parameter count:")
print(sum(p.numel() for p in decoder_instance.parameters() if p.requires_grad))

# Get sample
sample = next(iter(train_data_loader))

# Print sample stats
print(sample)

# Split sample
input_graph, input_mesh = sample

# Clone and run
input = input_graph.clone().to(device)
encoding = encoder_instance(input)
print(encoding.size())
result, _ = decoder_instance(encoding)
print(result)

# Extract and split encoding
position = encoding[:, 0, :, :]
normal = encoding[:, 1, :, :]
encoding_lvl0 = encoding[:, 2:64//3+2, :, :]
encoding_lvl1 = encoding[:, 64//3+2:64//3*2+2, :, :]
encoding_lvl2 = encoding[:, 64//3*2+2:64//3*2+128//3+2, :, :]
encoding_lvl3 = encoding[:, 64//3*2+128//3+2:, :, :]

figs = []

# Visualize input mesh, input graph, output points
figs.append(vis_utils.visualize_mesh_graph_in_points_out(input_graph, input_mesh, result))

# Visualize closeness in feature space at level 0
# Calculate closeness
encoding_lvl0_squeezed = encoding_lvl0.view(encoding_lvl0.size(0), encoding_lvl0.size(1) * encoding_lvl0.size(2), encoding_lvl0.size(3))
query = encoding_lvl0_squeezed[:, :, 0].unsqueeze(-1).expand(encoding_lvl0_squeezed.size())
dist = ((encoding_lvl0_squeezed - query) ** 2).sum(dim=1)
figs.append(vis_utils.visualize_closeness(position, dist))

# Visualize closeness in feature space at level 3
encoding_lvl3_squeezed = encoding_lvl3.view(encoding_lvl3.size(0), encoding_lvl3.size(1) * encoding_lvl3.size(2), encoding_lvl3.size(3))
query = encoding_lvl3_squeezed[:, :, 0].unsqueeze(-1).expand(encoding_lvl3_squeezed.size())
dist = ((encoding_lvl3_squeezed - query) ** 2).sum(dim=1)
figs.append(vis_utils.visualize_closeness(position, dist))

# Visualize closeness in scalar feature space at level 0
encoding_lvl0_squared_length = (encoding_lvl0 ** 2).sum(dim = 2)
query = encoding_lvl0_squared_length[:, :, 0].unsqueeze(-1).expand(encoding_lvl0_squared_length.size())
dist = ((encoding_lvl0_squared_length - query) ** 2).sum(dim = 1)
figs.append(vis_utils.visualize_closeness(position, dist))

plt.show()