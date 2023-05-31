import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv

class GNNVer(nn.Module):
    def __init__(self, device):
        super(GNNVer, self).__init__()
        self.device = device
        
        feu = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 16, 3]
        #initialize the layers
        #12GCN layers==> we have 12 message passing steps and then a linear output layer
        self.conv1  = GCNConv(feu[0], feu[1]) #number_features, hidden_channels or filters
        self.conv2  = GCNConv(feu[1], feu[2])
        self.conv3  = GCNConv(feu[2], feu[3])
        self.conv4  = GCNConv(feu[3], feu[4])
        self.conv5  = GCNConv(feu[4], feu[5])
        self.conv6  = GCNConv(feu[5], feu[6])
        self.conv7  = GCNConv(feu[6], feu[7])
        self.conv8  = GCNConv(feu[7], feu[8])
        self.conv9  = GCNConv(feu[8], feu[9])
        self.conv10 = GCNConv(feu[9], feu[10])
        self.conv11 = GCNConv(feu[10], feu[11])
        self.conv12 = GCNConv(feu[11], feu[12])
        #2 FCN layers, linear that output probability
        self.linear1 = nn.Linear(feu[12], feu[13]) #apply linear transformation to the imcoming data
        self.linear2 = nn.Linear(feu[13], feu[14])
    
        self.bn1 = nn.BatchNorm1d(feu[1])
        self.bn2 = nn.BatchNorm1d(feu[2])
        self.bn3 = nn.BatchNorm1d(feu[3])
        self.bn4 = nn.BatchNorm1d(feu[4])
        self.bn5 = nn.BatchNorm1d(feu[5])
        self.bn6 = nn.BatchNorm1d(feu[6])
        self.bn7 = nn.BatchNorm1d(feu[7])
        self.bn8 = nn.BatchNorm1d(feu[8])
        self.bn9 = nn.BatchNorm1d(feu[9])
        self.bn10 = nn.BatchNorm1d(feu[10])
        self.bn11 = nn.BatchNorm1d(feu[11])
        self.bn12 = nn.BatchNorm1d(feu[12])

        self.l_relu = nn.LeakyReLU()
            

    def forward(self, data):
        #z1: node features tensor, edge_index: graph connectivity
        z1, x_pos, edge_index = data.z1.to(self.device), data.x_pos.to(self.device), data.edge_index.to(self.device)
        # print(edge_index.shape)
        # print (edge_index.type())
        # n1 = torch.randn(x_pos.shape[0], x_pos.shape[1]).to(self.device) * 1e-5
        #first message passing layer (transformation) 16: is the node embedding size
        dx = self.l_relu(self.bn1(self.conv1(z1, edge_index)))
        dx = self.l_relu(self.bn2(self.conv2(dx, edge_index)))
        dx = self.l_relu(self.bn3(self.conv3(dx, edge_index)))
        dx = self.l_relu(self.bn4(self.conv4(dx, edge_index)))
        dx = self.l_relu(self.bn5(self.conv5(dx, edge_index)))
        dx = self.l_relu(self.bn6(self.conv6(dx, edge_index)))
        dx = self.l_relu(self.bn7(self.conv7(dx, edge_index)))
        dx = self.l_relu(self.bn8(self.conv8(dx, edge_index)))
        dx = self.l_relu(self.bn9(self.conv9(dx, edge_index)))
        dx = self.l_relu(self.bn10(self.conv10(dx, edge_index)))
        dx = self.l_relu(self.bn11(self.conv11(dx, edge_index)))
        dx = self.l_relu(self.bn12(self.conv12(dx, edge_index)))
        
        dx = self.l_relu(self.linear1(dx))
        dx = self.linear2(dx)
        
        return x_pos + dx

class GNNFac(nn.Module):
    def __init__(self, device):
        super(GNNFac, self).__init__()
        self.device = device
        
        feu = [7, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 16, 3]

        self.conv1  = GCNConv(feu[0], feu[1])
        self.conv2  = GCNConv(feu[1], feu[2])
        self.conv3  = GCNConv(feu[2], feu[3])
        self.conv4  = GCNConv(feu[3], feu[4])
        self.conv5  = GCNConv(feu[4], feu[5])
        self.conv6  = GCNConv(feu[5], feu[6])
        self.conv7  = GCNConv(feu[6], feu[7])
        self.conv8  = GCNConv(feu[7], feu[8])
        self.conv9  = GCNConv(feu[8], feu[9])
        self.conv10 = GCNConv(feu[9], feu[10])
        self.conv11 = GCNConv(feu[10], feu[11])
        self.conv12 = GCNConv(feu[11], feu[12])

        self.linear1 = nn.Linear(feu[12], feu[13]) #Applies a linear transformation to the incoming data
        self.linear2 = nn.Linear(feu[13], feu[14])
    
        self.bn1 = nn.BatchNorm1d(feu[1])
        self.bn2 = nn.BatchNorm1d(feu[2])
        self.bn3 = nn.BatchNorm1d(feu[3])
        self.bn4 = nn.BatchNorm1d(feu[4])
        self.bn5 = nn.BatchNorm1d(feu[5])
        self.bn6 = nn.BatchNorm1d(feu[6])
        self.bn7 = nn.BatchNorm1d(feu[7])
        self.bn8 = nn.BatchNorm1d(feu[8])
        self.bn9 = nn.BatchNorm1d(feu[9])
        self.bn10 = nn.BatchNorm1d(feu[10])
        self.bn11 = nn.BatchNorm1d(feu[11])
        self.bn12 = nn.BatchNorm1d(feu[12])

        self.l_relu = nn.LeakyReLU() #use ReLU as our intermediate non-linearity 
            

    def forward(self, data):
        # z1, x_pos, edge_index = data.z1, data.x_pos, data.edge_index
        # n1 = torch.randn(x_pos.shape[0], x_pos.shape[1]) * 1e-5

        z2, x_pos, face_index = data.z2.to(self.device), data.x_pos.to(self.device), data.face_index.to(self.device)
        # print (face_index.type())
        # n2 = torch.randn(z2.shape[0], z2.shape[1]).to(self.device) * 0.01
        dx = self.l_relu(self.bn1(self.conv1(z2, face_index)))
        dx = self.l_relu(self.bn2(self.conv2(dx, face_index)))
        dx = self.l_relu(self.bn3(self.conv3(dx, face_index)))
        dx = self.l_relu(self.bn4(self.conv4(dx, face_index)))
        dx = self.l_relu(self.bn5(self.conv5(dx, face_index)))
        dx = self.l_relu(self.bn6(self.conv6(dx, face_index)))
        dx = self.l_relu(self.bn7(self.conv7(dx, face_index)))
        dx = self.l_relu(self.bn8(self.conv8(dx, face_index)))
        dx = self.l_relu(self.bn9(self.conv9(dx, face_index)))
        dx = self.l_relu(self.bn10(self.conv10(dx, face_index)))
        dx = self.l_relu(self.bn11(self.conv11(dx, face_index)))
        dx = self.l_relu(self.bn12(self.conv12(dx, face_index)))
        
        dx = self.l_relu(self.linear1(dx))
        dx = torch.tanh(self.linear2(dx))
        #Returns the matrix norm or vector norm of a given tensor.
        dx_norm = torch.reciprocal(torch.norm(dx, dim=1, keepdim=True).expand(-1, 3) + 1.0e-12)#Returns a new tensor with the reciprocal of the elements of input
        x = torch.mul(dx, dx_norm) #Multiplies input by other
        return x


class Net(nn.Module):
    def __init__(self, flags):
        super(Net, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flags = flags
        feu = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 32, 3]
        if self.flags:
            # skip net
            self.conv1  = GCNConv(feu[0], feu[1])
            self.conv2  = GCNConv(feu[1], feu[2])
            self.conv3  = GCNConv(feu[2], feu[3])
            self.conv4  = GCNConv(feu[3], feu[4])
            self.conv5  = GCNConv(feu[4], feu[5])
            self.conv6  = GCNConv(feu[5], feu[6])
            self.conv7  = GCNConv(feu[6], feu[7])
            self.conv8  = GCNConv(feu[7]+feu[6], feu[8])
            self.conv9  = GCNConv(feu[8]+feu[5], feu[9])
            self.conv10 = GCNConv(feu[9]+feu[4], feu[10])
            self.conv11 = GCNConv(feu[10]+feu[3], feu[11])
            self.conv12 = GCNConv(feu[11]+feu[2], feu[12])
            self.conv13 = GCNConv(feu[12]+feu[1], feu[13])
            self.linear1 = nn.Linear(feu[13], feu[14])
            
            self.bn1 = nn.BatchNorm1d(feu[1])
            self.bn2 = nn.BatchNorm1d(feu[2])
            self.bn3 = nn.BatchNorm1d(feu[3])
            self.bn4 = nn.BatchNorm1d(feu[4])
            self.bn5 = nn.BatchNorm1d(feu[5])
            self.bn6 = nn.BatchNorm1d(feu[6])
            self.bn7 = nn.BatchNorm1d(feu[7])
            self.bn8 = nn.BatchNorm1d(feu[8])
            self.bn9 = nn.BatchNorm1d(feu[9])
            self.bn10 = nn.BatchNorm1d(feu[10])
            self.bn11 = nn.BatchNorm1d(feu[11])
            self.bn12 = nn.BatchNorm1d(feu[12])
            self.bn13 = nn.BatchNorm1d(feu[13])
            self.l_relu = nn.LeakyReLU()
        
        else:
            # normal net
            feu = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 32, 16, 3]
            self.conv1  = GCNConv(feu[0], feu[1])
            self.conv2  = GCNConv(feu[1], feu[2])
            self.conv3  = GCNConv(feu[2], feu[3])
            self.conv4  = GCNConv(feu[3], feu[4])
            self.conv5  = GCNConv(feu[4], feu[5])
            self.conv6  = GCNConv(feu[5], feu[6])
            self.conv7  = GCNConv(feu[6], feu[7])
            self.conv8  = GCNConv(feu[7], feu[8])
            self.conv9  = GCNConv(feu[8], feu[9])
            self.conv10 = GCNConv(feu[9], feu[10])
            self.conv11 = GCNConv(feu[10], feu[11])
            self.conv12 = GCNConv(feu[11], feu[12])
            self.conv13 = GCNConv(feu[12], feu[13])
            self.linear1 = nn.Linear(feu[13], feu[14])
            self.linear2 = nn.Linear(feu[14], feu[15])
            
            self.bn1 = nn.BatchNorm1d(feu[1])
            self.bn2 = nn.BatchNorm1d(feu[2])
            self.bn3 = nn.BatchNorm1d(feu[3])
            self.bn4 = nn.BatchNorm1d(feu[4])
            self.bn5 = nn.BatchNorm1d(feu[5])
            self.bn6 = nn.BatchNorm1d(feu[6])
            self.bn7 = nn.BatchNorm1d(feu[7])
            self.bn8 = nn.BatchNorm1d(feu[8])
            self.bn9 = nn.BatchNorm1d(feu[9])
            self.bn10 = nn.BatchNorm1d(feu[10])
            self.bn11 = nn.BatchNorm1d(feu[11])
            self.bn12 = nn.BatchNorm1d(feu[12])
            self.bn13 = nn.BatchNorm1d(feu[13])
            self.l_relu = nn.LeakyReLU()
            

    def forward(self, data):
        x = np.random.normal(0, 0.1, size=(data.x.shape[0], 16))
        x, edge_index, x_pos = data.x.to(self.device), data.edge_index.to(self.device), data.x_pos.to(self.device)
        
        if self.flags:
            # skip net
            dx = self.conv1(x, edge_index)
            dx = self.bn1(dx)
            dx = self.l_relu(dx)
            skip1 = dx

            dx = self.conv2(dx, edge_index)
            dx = self.bn2(dx)
            dx = self.l_relu(dx)
            skip2 = dx

            dx = self.conv3(dx, edge_index)
            dx = self.bn3(dx)
            dx = self.l_relu(dx)
            skip3 = dx

            dx = self.conv4(dx, edge_index)
            dx = self.bn4(dx)
            dx = self.l_relu(dx)
            skip4 = dx

            dx = self.conv5(dx, edge_index)
            dx = self.bn5(dx)
            dx = self.l_relu(dx)
            skip5 = dx
            
            dx = self.conv6(dx, edge_index)
            dx = self.bn6(dx)
            dx = self.l_relu(dx)
            skip6 = dx

            dx = self.conv7(dx, edge_index)
            dx = self.bn7(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip6], dim=1)
            dx = self.conv8(dx, edge_index)
            dx = self.bn8(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip5], dim=1)
            dx = self.conv9(dx, edge_index)
            dx = self.bn9(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip4], dim=1)
            dx = self.conv10(dx, edge_index)
            dx = self.bn10(dx)
            dx = self.l_relu(dx)
            
            dx = torch.cat([dx, skip3], dim=1)
            dx = self.conv11(dx, edge_index)
            dx = self.bn11(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip2], dim=1)
            dx = self.conv12(dx, edge_index)
            dx = self.bn12(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip1], dim=1)
            dx = self.conv13(dx, edge_index)
            dx = self.bn13(dx)
            dx = self.l_relu(dx)
            dx = self.linear1(dx)
        else:
            # normal net
            dx = self.conv1(x, edge_index)
            dx = self.bn1(dx)
            dx = self.l_relu(dx)

            dx = self.conv2(dx, edge_index)
            dx = self.bn2(dx)
            dx = self.l_relu(dx)

            dx = self.conv3(dx, edge_index)
            dx = self.bn3(dx)
            dx = self.l_relu(dx)

            dx = self.conv4(dx, edge_index)
            dx = self.bn4(dx)
            dx = self.l_relu(dx)

            dx = self.conv5(dx, edge_index)
            dx = self.bn5(dx)
            dx = self.l_relu(dx)
            
            dx = self.conv6(dx, edge_index)
            dx = self.bn6(dx)
            dx = self.l_relu(dx)

            dx = self.conv7(dx, edge_index)
            dx = self.bn7(dx)
            dx = self.l_relu(dx)

            dx = self.conv8(dx, edge_index)
            dx = self.bn8(dx)
            dx = self.l_relu(dx)

            dx = self.conv9(dx, edge_index)
            dx = self.bn9(dx)
            dx = self.l_relu(dx)

            dx = self.conv10(dx, edge_index)
            dx = self.bn10(dx)
            dx = self.l_relu(dx)

            dx = self.conv11(dx, edge_index)
            dx = self.bn11(dx)
            dx = self.l_relu(dx)
            
            dx = self.conv12(dx, edge_index)
            dx = self.bn12(dx)
            dx = self.l_relu(dx)
            
            dx = self.conv13(dx, edge_index)
            dx = self.bn13(dx)
            dx = self.l_relu(dx)
            
            dx = self.linear1(dx)
            dx = self.l_relu(dx)
            dx = self.linear2(dx)
        
        return x_pos + dx


class Net1(nn.Module):
    def __init__(self, device):
        super(Net1, self).__init__()
        self.device = device
        feu = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 32, 3]     
        # feu = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 16, 3]

        self.conv1  = GCNConv(feu[0], feu[1])
        self.conv2  = GCNConv(feu[1], feu[2])
        self.conv3  = GCNConv(feu[2], feu[3])
        self.conv4  = GCNConv(feu[3], feu[4])
        self.conv5  = GCNConv(feu[4], feu[5])
        self.conv6  = GCNConv(feu[5], feu[6])
        self.conv7  = GCNConv(feu[6], feu[7])
        self.conv8  = GCNConv(feu[7]+feu[6], feu[8])
        self.conv9  = GCNConv(feu[8]+feu[5], feu[9])
        self.conv10 = GCNConv(feu[9]+feu[4], feu[10])
        self.conv11 = GCNConv(feu[10]+feu[3], feu[11])
        self.conv12 = GCNConv(feu[11]+feu[2], feu[12])
        self.conv13 = GCNConv(feu[12]+feu[1], feu[13])
        self.linear1 = nn.Linear(feu[13], feu[14])
        
        self.bn1 = nn.BatchNorm1d(feu[1])
        self.bn2 = nn.BatchNorm1d(feu[2])
        self.bn3 = nn.BatchNorm1d(feu[3])
        self.bn4 = nn.BatchNorm1d(feu[4])
        self.bn5 = nn.BatchNorm1d(feu[5])
        self.bn6 = nn.BatchNorm1d(feu[6])
        self.bn7 = nn.BatchNorm1d(feu[7])
        self.bn8 = nn.BatchNorm1d(feu[8])
        self.bn9 = nn.BatchNorm1d(feu[9])
        self.bn10 = nn.BatchNorm1d(feu[10])
        self.bn11 = nn.BatchNorm1d(feu[11])
        self.bn12 = nn.BatchNorm1d(feu[12])
        self.bn13 = nn.BatchNorm1d(feu[13])
        self.l_relu = nn.LeakyReLU()
                       
    def forward(self, data):
        #z1: node features tensor, edge_index: graph connectivity
        z1, x_pos, edge_index = data.z1.to(self.device), data.x_pos.to(self.device), data.edge_index.to(self.device)

        dx = self.l_relu(self.bn1(self.conv1(z1, edge_index)))
        skip1 = dx
        dx = self.l_relu(self.bn2(self.conv2(dx, edge_index)))
        skip2 = dx
        dx = self.l_relu(self.bn3(self.conv3(dx, edge_index)))
        skip3 = dx
        dx = self.l_relu(self.bn4(self.conv4(dx, edge_index)))
        skip4 = dx
        dx = self.l_relu(self.bn5(self.conv5(dx, edge_index)))
        skip5 = dx
        dx = self.l_relu(self.bn6(self.conv6(dx, edge_index)))
        skip6 = dx
        dx = self.l_relu(self.bn7(self.conv7(dx, edge_index)))
        dx = torch.cat([dx, skip6], dim=1)
        dx = self.l_relu(self.bn8(self.conv8(dx, edge_index)))
        dx = torch.cat([dx, skip5], dim=1)
        dx = self.l_relu(self.bn9(self.conv9(dx, edge_index)))
        dx = torch.cat([dx, skip4], dim=1)
        dx = self.l_relu(self.bn10(self.conv10(dx, edge_index)))
        dx = torch.cat([dx, skip3], dim=1)
        dx = self.l_relu(self.bn11(self.conv11(dx, edge_index)))
        dx = torch.cat([dx, skip2], dim=1)
        dx = self.l_relu(self.bn12(self.conv12(dx, edge_index)))
        dx = torch.cat([dx, skip1], dim=1)
        dx = self.l_relu(self.bn13(self.conv13(dx, edge_index)))
        dx = self.l_relu(self.linear1(dx))
        dx = self.linear2(dx)
        
        
        return x_pos + dx


