#This file can replace for the networks.py if the input mesh is too large.
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNVer(nn.Module):
    def __init__(self, device):
        super(GNNVer, self).__init__()
        self.device = device
        
        feu = [16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 3]

        self.conv1  = GCNConv(feu[0], feu[1])
        self.conv2  = GCNConv(feu[1], feu[2])
        self.conv3  = GCNConv(feu[2], feu[3])
        self.conv4  = GCNConv(feu[3], feu[4])
        self.conv5  = GCNConv(feu[4], feu[5])
        self.conv6  = GCNConv(feu[5], feu[6])
        self.conv7  = GCNConv(feu[6], feu[7])
        self.conv8  = GCNConv(feu[7], feu[8])

        self.linear1 = nn.Linear(feu[8], feu[9]) #apply linear transformation to the imcoming data
        self.linear2 = nn.Linear(feu[9], feu[10])
    
        self.bn1 = nn.BatchNorm1d(feu[1])
        self.bn2 = nn.BatchNorm1d(feu[2])
        self.bn3 = nn.BatchNorm1d(feu[3])
        self.bn4 = nn.BatchNorm1d(feu[4])
        self.bn5 = nn.BatchNorm1d(feu[5])
        self.bn6 = nn.BatchNorm1d(feu[6])
        self.bn7 = nn.BatchNorm1d(feu[7])
        self.bn8 = nn.BatchNorm1d(feu[8])

        self.l_relu = nn.LeakyReLU()
            

    def forward(self, data):
        z1, x_pos, edge_index = data.z1.to(self.device), data.x_pos.to(self.device), data.edge_index.to(self.device)
        dx = self.l_relu(self.bn1(self.conv1(z1, edge_index)))
        dx = self.l_relu(self.bn2(self.conv2(dx, edge_index)))
        dx = self.l_relu(self.bn3(self.conv3(dx, edge_index)))
        dx = self.l_relu(self.bn4(self.conv4(dx, edge_index)))
        dx = self.l_relu(self.bn5(self.conv5(dx, edge_index)))
        dx = self.l_relu(self.bn6(self.conv6(dx, edge_index)))
        dx = self.l_relu(self.bn7(self.conv7(dx, edge_index)))
        dx = self.l_relu(self.bn8(self.conv8(dx, edge_index)))
        
        dx = self.l_relu(self.linear1(dx))
        dx = self.linear2(dx)
        
        return x_pos + dx

class GNNFac(nn.Module):
    def __init__(self, device):
        super(GNNFac, self).__init__()
        self.device = device
        
        feu = [7, 32, 64, 128, 256,  256, 128, 64, 32, 16, 3]

        self.conv1  = GCNConv(feu[0], feu[1])
        self.conv2  = GCNConv(feu[1], feu[2])
        self.conv3  = GCNConv(feu[2], feu[3])
        self.conv4  = GCNConv(feu[3], feu[4])
        self.conv5  = GCNConv(feu[4], feu[5])
        self.conv6  = GCNConv(feu[5], feu[6])
        self.conv7  = GCNConv(feu[6], feu[7])
        self.conv8  = GCNConv(feu[7], feu[8])

        self.linear1 = nn.Linear(feu[8], feu[9]) #Applies a linear transformation to the incoming data
        self.linear2 = nn.Linear(feu[9], feu[10])
    
        self.bn1 = nn.BatchNorm1d(feu[1])
        self.bn2 = nn.BatchNorm1d(feu[2])
        self.bn3 = nn.BatchNorm1d(feu[3])
        self.bn4 = nn.BatchNorm1d(feu[4])
        self.bn5 = nn.BatchNorm1d(feu[5])
        self.bn6 = nn.BatchNorm1d(feu[6])
        self.bn7 = nn.BatchNorm1d(feu[7])
        self.bn8 = nn.BatchNorm1d(feu[8])


        self.l_relu = nn.LeakyReLU() #use ReLU as our intermediate non-linearity 
            

    def forward(self, data):
        z2, x_pos, face_index = data.z2.to(self.device), data.x_pos.to(self.device), data.face_index.to(self.device)

        dx = self.l_relu(self.bn1(self.conv1(z2, face_index)))
        dx = self.l_relu(self.bn2(self.conv2(dx, face_index)))
        dx = self.l_relu(self.bn3(self.conv3(dx, face_index)))
        dx = self.l_relu(self.bn4(self.conv4(dx, face_index)))
        dx = self.l_relu(self.bn5(self.conv5(dx, face_index)))
        dx = self.l_relu(self.bn6(self.conv6(dx, face_index)))
        dx = self.l_relu(self.bn7(self.conv7(dx, face_index)))
        dx = self.l_relu(self.bn8(self.conv8(dx, face_index)))

        
        dx = self.l_relu(self.linear1(dx))
        dx = torch.tanh(self.linear2(dx))
        dx_norm = torch.reciprocal(torch.norm(dx, dim=1, keepdim=True).expand(-1, 3) + 1.0e-12)#Returns a new tensor with the reciprocal of the elements of input
        x = torch.mul(dx, dx_norm) 
        return x