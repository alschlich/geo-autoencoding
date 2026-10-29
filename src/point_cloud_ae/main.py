import numpy as np
import time
import torch
import torch.optim as optim
from itertools import chain
import models.regressor as regressor
import models.encoder as encoder
import models.decoder as decoder
import networks
import data_loaders
import output_utils
import losses

# Detect CUDA-device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set parameters
input_node_count = 2048
output_node_count = input_node_count
encoding_size = 1024
epoch_count = 250
batch_size = 14 if torch.cuda.is_available() else 3
train_set_percentage = 0.9
learning_rate = 0.001
output_path = "output/"
data_input_path = "data/dataset.pkl"

# Set anomaly detection
torch.autograd.set_detect_anomaly(False)

# Clear output direcory
# output_utils.clear_path(output_path)

# Make autoencoder network
encoder = encoder.VNGEncoder(encoding_size)
encoder.to(device)
decoder = decoder.VNGDecoder(encoding_size, output_node_count)
decoder.to(device)

# Make optimizer
optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr = learning_rate)

# Load data
train_data_loader, test_data_loader = data_loaders.get_graph_dataloaders(data_input_path, batch_size, train_set_percentage)

# Build network structure
networks = networks.Networks(encoder, decoder, optimizer, device)

# Lists to save losses
train_loss_list = []
test_loss_list = []

# Train and overwrite best model
for epoch_idx in range(epoch_count):
    startTime = time.time()
    
    train_loss = networks.train_epoch(train_data_loader)
    train_loss_list.append(train_loss)
    
    test_loss = networks.test_epoch(test_data_loader)
    test_loss_list.append(test_loss)
    
    epoch_time = time.time() - startTime
    print("Epoch time | Epoch test loss | Epoch train loss")
    print(repr(epoch_time) + "  " + repr(test_loss) + "  " + repr(train_loss))
    
    if (test_loss <= min(test_loss_list)):
        networks.save(output_path)