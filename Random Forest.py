import requests
import os

from dataset import HW3Dataset
from torch_geometric.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]
print(data)

# Get test and train data:
X_train = np.array(data.x[data.train_mask])
X_test = np.array(data.x[data.val_mask])


Y_train = np.array(data.y[data.train_mask])
Y_test = np.array(data.y[data.val_mask])

# Random Forest Classifier:
clf = RandomForestClassifier(n_estimators = 250, max_depth=1000)

clf.fit(X_train, Y_train.ravel())

print(clf.score(X_test, Y_test.ravel()))