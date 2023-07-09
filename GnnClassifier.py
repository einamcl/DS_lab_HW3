# Define nn:

from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_0, hidden_dim_1,hidden_dim_2, output_dim):
        super(GNN, self).__init__()

        torch.manual_seed(42)
        self.conv0 = GCNConv(input_dim, hidden_dim_0).to(device)
        self.conv1 = GCNConv(hidden_dim_0, hidden_dim_1).to(device)
        self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2).to(device)

        self.fc = nn.Linear(hidden_dim_2, output_dim).to(device)

    def forward(self, x, edge_index):
        x= self.conv0(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        x = self.fc(x)
        # x = self.conv3(x, edge_index)
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)

        return F.log_softmax(x, dim=1)
