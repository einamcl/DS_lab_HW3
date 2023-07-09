import requests
import os
# !pip install torch_geometric
from torch_geometric.data import Dataset
import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_geometric.data import Data

from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

from GnnClassifier import GNN
from dataset import HW3Dataset


def get_predictions(model, x, edge_index):
    model.eval()
    with torch.no_grad():
        test_output = model(x, edge_index)
        _, predicted_labels = test_output.max(dim=1)

    predicted_labels = predicted_labels.cpu()
    return predicted_labels.numpy()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    # Define the device to be used (CPU or GPU if available)

    # Define the dimensions of your input and output

    input_dim = 128  # Dimensionality of input features (x)
    hidden_dim_0 = 300  # Dimensionality of hidden layer
    hidden_dim_1 = 250
    hidden_dim_2 = 180
    output_dim = 40  # Number of classes in your classification task

    # Create the model
    model = GNN(input_dim, hidden_dim_0, hidden_dim_1, hidden_dim_2, output_dim).to(device)
    model = torch.load('model.pt')
    # Define the loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = torch.load('data/hw3/processed/data.pt').to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    print(get_predictions(model, x, edge_index))
