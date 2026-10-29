import torch
import torch.nn as nn
import losses
from geom_utils import *

class Networks(nn.Module):
    def __init__(self, encoder, decoder, optimizer, device):
        super(Networks, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = device
    
    def forward (self, mode, x):
        assert mode in ["training_testing", "inference"]
        if mode == "training_testing":
            return self.forward_training_testing(x)
        elif mode == "inference":
            return self.forward_inference(x)

    def forward_training_testing (self, data):
        graph = data[0]
        mesh = data[1]
        graph = graph.to(self.device)
        mesh = mesh.to(self.device)
        x = graph
        self.train()
        # Assume data is centered and scaled down to sphere with r = 0.8
        # Record losses
        losses_dict = {}
        # Create training and testing data by random rotation
        x_1, transform_1 = rotate_graph_random(x)
        #x_2, transform_2 = rotate_graph_random(x)
        # Encode
        encoding_1 = self.encoder(x_1)
        #encoding_2, intermediate_losses_2 = self.encoder(x_2)
        # Calculate equivariance loss (should be zero if network is rotation equivariant)
        #equivariance_loss = losses.equivariance_loss(encoding_1, encoding_2, transform_2)
        #equivariance_loss.requires_grad = True
        #losses_dict["equivariance_loss"] = equivariance_loss
        # Decode
        y_1, pairwise_distance_loss = self.decoder(encoding_1)
        losses_dict["pairwise_distance_loss"] = pairwise_distance_loss
        #y_2 = self.decoder(encoding_2)
        #losses_dict["spreading_loss"] = spreading_loss
        # Calculate mesh distance loss
        mesh_distance_loss = losses.mesh_distance_loss(y_1, mesh)
        losses_dict["mesh_distance_loss"] = mesh_distance_loss
        #chamfer_loss_1 = losses.chamfer_loss(y_1, x_1)
        #chamfer_loss_2 = losses.chamfer_loss(x_2, y_2)
        #chamfer_loss = chamfer_loss_1 #+ chamfer_loss_2
        #losses_dict["chamfer_loss_1"] = chamfer_loss
        return y_1, encoding_1, losses_dict

    def forward_inference (self, x):
        self.eval()
        # Assume data is centered and scaled down to sphere with r = 0.8
        # Encode
        encoding = self.encoder(x)
        # Decode
        y = self.decoder(encoding)
        return y, encoding, keypoints

    def test_epoch(self, test_loader):
        epoch_loss = 0
        for data in test_loader:
            output_reconstructed, keypoints, losses_dict = self("training_testing", data)
            loss = sum(losses_dict.values())
            epoch_loss += loss.detach()
        return epoch_loss

    def train_epoch(self, train_loader):
        epoch_loss = 0
        for data in train_loader:
            self.optimizer.zero_grad()
            output_reconstructed, keypoints, losses_dict = self("training_testing", data)
            loss = sum(losses_dict.values())
            loss.backward()
            if not self.check_nan_gradient():
                self.optimizer.step()
            if str(self.device) == "cpu":
                print("Losses:")
                for key in losses_dict.keys():
                    print(f"{key}: {losses_dict[key]:.4f}")
            epoch_loss += loss.detach()
        return epoch_loss

    def save(self, path):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict()},
            path + "best_model.pth")
    
    def check_nan_gradient(self):
        for param in self.encoder.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).sum() > 0:
                    return True
        for param in self.decoder.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).sum() > 0:
                    return True            
        return False 