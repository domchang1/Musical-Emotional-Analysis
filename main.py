from unicodedata import bidirectional
import torch
import torch.nn.functional as F
import torch.nn as nn
import RNN
from pathlib import Path
import pandas as pd
import pickle
import numpy as np


def prediction_accuracy(output, label):
    output = (output > 0.5).long()
    return torch.eq(output, label).long()
    
#venv\Scripts\activate.bat
inputs_dst = Path(f"./inputs.pkl")
labels_dst = Path(f"./labels.pkl")
inputs = pd.read_pickle(inputs_dst)
labels = pd.read_pickle(labels_dst)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(inputs)
# print(labels)

input = torch.tensor(inputs[0], dtype=torch.float32)
# input = torch.unsqueeze(input, 0)
# print(input.shape)
model = RNN.RNN(input.shape[1], len(labels[0][0]), 20).to(device)
#test = torch.randn(3, 10)
# output = model(input)
# print(output)
# print(output.shape)
criterion = nn.BCEWithLogitsLoss() #pos_weight=torch.ones([32])
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    curr_loss = 0.0
    # counter = 0
    overall_accuracies = torch.zeros(32).to(device)
    for features, label in zip(inputs, labels):
        optimizer.zero_grad()
        data = torch.tensor(features, dtype=torch.float32).to(device)
        # print(data.shape)
        output = model(data)
        output = torch.squeeze(output)
        loss = criterion(output, torch.tensor(label[1]).to(device))
        loss = torch.inner(loss, torch.tensor(label[0]).to(device))
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()
        # print(optimizer)
        curr_loss += loss.item()
        # print(loss.item())
        overall_accuracies += prediction_accuracy(output, torch.tensor(label[1]).to(device))
    print('Epoch #%d Loss %.3f' % (epoch, curr_loss ))
    print(overall_accuracies / len(inputs))
    


