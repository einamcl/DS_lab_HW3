# !pip install torch_geometric
import pandas as pd
import torch
import torch.nn as nn
from GnnClassifier import GNN
from dataset import HW3Dataset


def get_predictions(model, x, edge_index):
    model.eval()
    with torch.no_grad():
        test_output = model(x, edge_index)
        _, predicted_labels = test_output.max(dim=1)

    predicted_labels = predicted_labels.cpu()
    return predicted_labels.numpy()


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
    model = GNN(input_dim, hidden_dim_0, hidden_dim_1, hidden_dim_2, output_dim)  # .to(device)
    model.load_state_dict(torch.load('model_GNN.pth', map_location=device))

    # Define the loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = torch.load('data/hw3/processed/data.pt').to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    print(get_predictions(model, x, edge_index))

    # save to csv
    predicted_labels = get_predictions(model, x, edge_index)
    indx = [i for i in range(data.x.shape[0])]
    predicts = pd.DataFrame()
    predicts['idx'] = indx
    predicts["prediction"] = predicted_labels
    predicts.to_csv("predict.csv")


    try:

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        train_mask = data.train_mask.to(device)
        val_mask = data.val_mask.to(device)

        accuracy = test(model)
        print(f"Accuracy: {accuracy:.2f}%")
    except:
        print()
