from unicodedata import bidirectional
import torch
import torch.nn.functional as F
import torch.nn as nn
import RNN
from pathlib import Path
import pandas as pd
import pickle
import numpy as np

#venv\Scripts\activate.bat
inputs_dst = Path(f"./inputs.pkl")
labels_dst = Path(f"./labels.pkl")
inputs = pd.read_pickle(inputs_dst)
labels = pd.read_pickle(labels_dst)
# print(inputs)
# print(labels)

input = torch.tensor(inputs[0], dtype=torch.float32)
# input = torch.unsqueeze(input, 0)
print(input.shape)
model = RNN.RNN(input.shape[1], len(labels[0][0]), 100)
#test = torch.randn(3, 10)
output = model(input)
print(output)
print(output.shape)
criterion = nn.BCEWithLogitsLoss() #check criteria
optimizer = torch.optim.Adam()

for epoch in range(100):
    for input, label in zip(inputs, labels):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label[1])
        loss = np.inner(loss, label[0])
        loss.backward()
        optimizer.step()
    

