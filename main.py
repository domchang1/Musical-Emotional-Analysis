from unicodedata import bidirectional
import torch
import torch.nn.functional as F
import torch.nn as nn
import RNN
import LSTM
from pathlib import Path
import pandas as pd
import pickle
import numpy as np


def prediction_accuracy(output, label, mask):
    output = (output > 0.5).long()
    return torch.eq(output, label).long() * mask

# results_dst = Path(f"./RNNresults.pkl")
# print(pd.read_pickle(results_dst))
# exit()
#venv\Scripts\activate.bat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
# model = torch.load('./model.pt').to(device)
# print(model)
# exit()
order = [i for i in range(502)]
np.random.shuffle(order)
# print(order)
inputs_dst = Path(f"./inputs.pkl")
labels_dst = Path(f"./labels.pkl")
inputs = pd.read_pickle(inputs_dst)
labels = pd.read_pickle(labels_dst)
new_inputs = []
new_labels = []
for i in order:
    new_inputs.append(inputs[i])
    new_labels.append(labels[i])
inputs = new_inputs
labels = new_labels

train_inputs = inputs[:351]
train_labels = labels[:351]
test_inputs = inputs[351:]
test_labels = labels[351:]

input = torch.tensor(inputs[0], dtype=torch.float32)
model = LSTM.LSTM(input.shape[1], len(labels[0][0]), 20).to(device)
criterion = nn.BCEWithLogitsLoss() #pos_weight=torch.ones([32])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
recorded_losses = []
recorded_accuracies = []

for epoch in range(50):
    curr_loss = 0.0
    model.train()
    overall_accuracies = torch.zeros(32).to(device)
    for features, label in zip(train_inputs, train_labels):
        optimizer.zero_grad()
        data = torch.tensor(features, dtype=torch.float32).to(device)
        # print(data.shape)
        output = model(data)
        output = torch.squeeze(output)
        loss = criterion(output, torch.tensor(label[1]).to(device))
        loss = torch.inner(loss, torch.tensor(label[0]).to(device))
        loss = torch.sum(loss)
        curr_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    model.eval()
    masks = torch.zeros(32).to(device)
    for features, label in zip(train_inputs, train_labels):
        data = torch.tensor(features, dtype=torch.float32).to(device)
        output = model(data)
        output = torch.squeeze(output)
        masks += torch.tensor(label[0]).to(device)
        overall_accuracies += prediction_accuracy(output, torch.tensor(label[1]).to(device), torch.tensor(label[0]).to(device))
    print('Epoch #%d Loss %.3f' % (epoch, curr_loss))
    recorded_losses.append(curr_loss)
    overall_accuracies /= masks
    print(overall_accuracies)
    recorded_accuracies.append(overall_accuracies)

model.eval()    
masks = torch.zeros(32).to(device)
train_accuracies = torch.zeros(32).to(device)
for features, label in zip(train_inputs, train_labels):
    data = torch.tensor(features, dtype=torch.float32).to(device)
    output = model(data)
    output = torch.squeeze(output)
    masks += torch.tensor(label[0]).to(device)
    train_accuracies += prediction_accuracy(output, torch.tensor(label[1]).to(device), torch.tensor(label[0]).to(device))

print(train_accuracies)
print(masks)
train_accuracies = torch.div(train_accuracies, masks)
# print("Final Accuracy: " + test_accuracies)
#torch.save(model,'model.pt')
print(train_accuracies)


masks = torch.zeros(32).to(device)
test_accuracies = torch.zeros(32).to(device)
for features, label in zip(test_inputs, test_labels):
    data = torch.tensor(features, dtype=torch.float32).to(device)
    output = model(data)
    output = torch.squeeze(output)
    masks += torch.tensor(label[0]).to(device)
    test_accuracies += prediction_accuracy(output, torch.tensor(label[1]).to(device), torch.tensor(label[0]).to(device))

print(test_accuracies)
print(masks)
test_accuracies = torch.div(test_accuracies, masks)
# print("Final Accuracy: " + test_accuracies)
torch.save(model,'LSTMmodel.pt')
print(test_accuracies)
results = {"Train Loss": recorded_losses, "Train Accuracies": recorded_accuracies, "Test Accuracy": test_accuracies}
results_dst = Path(f"./LSTMresults.pkl")
results_dst.write_bytes(pickle.dumps(results))



