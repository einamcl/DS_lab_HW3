from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from GnnClassifier import GNN
from torch_geometric.data import Dataset
from dataset import HW3Dataset


# define Training proccess
def train(model):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(x, edge_index)  # Perform a single forward pass.
    loss = criterion(out[train_mask],
                     data.y[train_mask].squeeze())  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss.item()


def test(model):
    model.eval()
    with torch.no_grad():
        test_output = model(x, edge_index)[val_mask]
        _, predicted_labels = test_output.max(dim=1)

    predicted_labels = predicted_labels.cpu()
    ground_truth_labels = y[val_mask].squeeze().cpu()

    correct = (predicted_labels == ground_truth_labels).sum().item()
    total = ground_truth_labels.size(0)
    accuracy = correct / total * 100
    return accuracy


def get_predictions(model):
    model.eval()
    with torch.no_grad():
        test_output = model(x, edge_index)
        _, predicted_labels = test_output.max(dim=1)

    predicted_labels = predicted_labels.cpu()
    return predicted_labels.numpy()


def save_model(model):
    torch.save(model.state_dict(), 'model_GNN.pth')


def load_model():
    model = GNN(input_dim, hidden_dim_0, hidden_dim_1, hidden_dim_2,
                output_dim)  # Create the model
    model.load_state_dict(torch.load('model_GNN.pth'))
    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Import data
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]

    # Define the device to be used (CPU or GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the dimensions of your input and output
    input_dim = 128  # Dimensionality of input features (x)
    hidden_dim_0 = 300  # Dimensionality of hidden layer
    hidden_dim_1 = 250
    hidden_dim_2 = 180
    output_dim = 40  # Number of classes in your classification task
    num_epochs = 500

    # Create the model
    model = GNN(input_dim, hidden_dim_0, hidden_dim_1, hidden_dim_2, output_dim).to(device)

    # Define the loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = torch.load('data/hw3/processed/data.pt').to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    # Run training and testing:

    accuracies = []
    losses = []

    # Train model:
    for epoch in range(1, num_epochs):
        if epoch == 500:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if epoch == 3000:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        loss = train(model)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        losses.append(loss)

        accuracy = test(model)
        print(f"Accuracy: {accuracy:.2f}%")
        accuracies.append(accuracy)

    # Save the model
    save_model(model)

    # Plot the testing accuracies and losses
    plt.plot(range(len(accuracies)), accuracies)
    plt.title(f"Accuracy per Epoch with hidden sizes of: {hidden_dim_0},{hidden_dim_1},{hidden_dim_2}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.show()

    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss per Epoch with hidden sizes of: {hidden_dim_0},{hidden_dim_1},{hidden_dim_2}")
    plt.show()

    # Save Predictions to csv file:
    predicted_labels = get_predictions(model)
    indx = [i for i in range(data.x.shape[0])]
    predicts = pd.DataFrame()
    predicts['idx'] = indx
    predicts["prediction"] = predicted_labels
    predicts.to_csv("predict.csv")
