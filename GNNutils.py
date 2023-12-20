import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


    
class part_seg1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(part_seg1, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        # CNN layers
        self.cnn1 = nn.Conv1d(input_dim, 32, kernel_size=1)
        self.cnn2 = nn.Conv1d(32, 64, kernel_size=1)
        self.cnn3 = nn.Conv1d(64, 128, kernel_size=1)
        
        # GCN layers
        self.gcn1 = GCNConv(128, 256)
        self.gcn2 = GCNConv(256, 512)
        self.gcn3 = GCNConv(512, 1024)

        # CNN layers
        self.cnnd1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.cnnd2 = nn.Conv1d(512, 256, kernel_size=1)
        self.cnnd3 = nn.Conv1d(256, 128, kernel_size=1)
        self.cnnd4 = nn.Conv1d(128, output_dim, kernel_size=1)
        
        
    def forward(self, x, edge_index=None):
        # CNN forward pass
        x = x.view(1, x.size(0), x.size(1))
        x = x.transpose(2,1)  # Reshape input to (batch_size, 6, n)
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        
        # GCN forward pass
        x = x.transpose(2,1)  # Reshape back to (batch_size, n, 128)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))
        # x = self.gcn3(x, edge_index)

        x = x.transpose(2,1)  # Reshape input to (batch_size, 6, n)
        x = F.relu(self.cnnd1(x))
        x = F.relu(self.cnnd2(x))
        x = F.relu(self.cnnd3(x))
        x = self.cnnd4(x)
        x = x.transpose(2,1)
        x = torch.softmax(x.view(-1, self.output_dim), dim=1)
        
        return x
    

class part_seg2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(part_seg2, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        # CNN layers
        self.cnn1 = nn.Conv1d(input_dim, 32, kernel_size=1)
        self.cnn2 = nn.Conv1d(32, 64, kernel_size=1)
        self.cnn3 = nn.Conv1d(64, 128, kernel_size=1)
        
        # GCN layers
        self.gcn1 = GCNConv(128, 256)
        self.gcn2 = GCNConv(256, 512)
        self.gcn3 = GCNConv(512, output_dim)
        
    def forward(self, x, edge_index=None):
        # CNN forward pass
        x = x.view(1, x.size(0), x.size(1))
        x = x.transpose(2,1)  # Reshape input to (batch_size, 6, n)
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        
        # GCN forward pass
        x = x.transpose(2,1)  # Reshape back to (batch_size, n, 128)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        # x = F.relu(self.gcn3(x, edge_index))
        x = self.gcn3(x, edge_index)
        x = torch.softmax(x.view(-1, self.output_dim), dim=1)
        
        return x


class part_seg3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(part_seg3, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop = nn.Dropout(0.3)
        # CNN layers
        self.cnn1 = nn.Conv1d(input_dim, 32, kernel_size=1)
        self.cnn2 = nn.Conv1d(32, 64, kernel_size=1)
        self.cnn3 = nn.Conv1d(64, 128, kernel_size=1)
        
        # GCN layers
        self.gcn1 = GCNConv(128, 256)
        self.gcn2 = GCNConv(256, 512)
        self.gcn3 = GCNConv(512, 1024)

        # CNN layers
        self.cnnd1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.cnnd2 = nn.Conv1d(512, 256, kernel_size=1)
        self.cnnd3 = nn.Conv1d(256, 128, kernel_size=1)
        self.cnnd4 = nn.Conv1d(128, output_dim, kernel_size=1)
        
        
    def forward(self, x, edge_index=None):
        # CNN forward pass
        x = x.view(1, x.size(0), x.size(1))
        x = x.transpose(2,1)  # Reshape input to (batch_size, 6, n)
        x = F.relu(self.cnn1(x))
        x = self.drop(x)
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        
        # GCN forward pass
        x = x.transpose(2,1)  # Reshape back to (batch_size, n, 128)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))
        # x = self.gcn3(x, edge_index)

        x = x.transpose(2,1)  # Reshape input to (batch_size, 6, n)
        x = F.relu(self.cnnd1(x))
        x = self.drop(x)
        x = F.relu(self.cnnd2(x))
        x = self.drop(x)
        x = F.relu(self.cnnd3(x))
        x = self.cnnd4(x)
        x = x.transpose(2,1)

        x = torch.softmax(x.view(-1, self.output_dim), dim=1)
        return x



def normalize_tensor(points):
    points = torch.tensor(points)
    points = (points - points.min()) / (points.max() - points.min()) * 2 - 1
    points = points.detach().numpy()
    return points
