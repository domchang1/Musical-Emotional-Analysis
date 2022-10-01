import torch
import torch.nn.functional as F
import torch.nn as nn
import RNN
import LSTM
from pathlib import Path
import pandas as pd
import pickle
import numpy as np

def prediction_accuracy(output, label):
    # output = (output >= 0.0).long()
    return torch.eq(output, label).long() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = pd.read_csv('train.csv', delimiter=',', encoding='latin-1')
dataset.fillna(0,inplace=True)
dataset.drop('Artist Name', inplace=True, axis=1)
dataset.drop('Track Name', inplace=True, axis=1)
dataset.drop('Popularity', inplace=True, axis=1)
dataset.drop('duration_in min/ms', inplace=True, axis=1)
dataset.columns = ['danceability', 'energy', 
'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
 'liveness', 'valence', 'tempo', 'time_signature', 'Class']
features = dataset[['danceability', 'energy', 
'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
 'liveness', 'valence', 'tempo', 'time_signature']].to_numpy()
classes = np.squeeze(dataset[['Class']].to_numpy())

# print(features.shape)
# print(classes)

train_inputs = features[:12597]
train_labels = classes[:12597]
test_inputs = features[12597:]
test_labels = classes[12597:]


model = LSTM.LSTM(256, 1, 128).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters())
recorded_losses = []
recorded_accuracies = []
overall_accuracies = 0
model.eval()

curr_loss = 0
for feature, label in zip(train_inputs, train_labels):
    feature = torch.tensor(feature, dtype=torch.float32).to(device)
    output = model(feature)
    # output = torch.squeeze(output)
    actual = torch.tensor(label, dtype=torch.float32).to(device)
    loss = criterion(output, torch.tensor(actual))
    # loss = torch.inner(loss, torch.tensor(label).to(device))
    loss = torch.sum(loss)
    curr_loss += loss.item()
    overall_accuracies += prediction_accuracy(output, torch.tensor(label).to(device))
print('Loss %.3f' % (curr_loss))

# print(overall_accuracies)
  

for epoch in range(50):
    curr_loss = 0.0
    model.train()
    overall_accuracies = torch.zeros(7).to(device)
    for feature, label in zip(train_inputs, train_labels):
        optimizer.zero_grad()
        feature = torch.tensor(feature, dtype=torch.float32).to(device)
        output = model(feature)
        output = torch.squeeze(output)
        # print(output)
        actual = torch.tensor(label, dtype=torch.float32).to(device)
        # print(actual)
        loss = criterion(output, actual)
        # loss = torch.inner(loss, torch.tensor(label).to(device))
        loss = torch.sum(loss)
        curr_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    model.eval()
    masks = torch.zeros(7).to(device)
    for feature, label in zip(train_inputs, train_labels):
        feature = torch.tensor(feature, dtype=torch.float32).to(device)
        output = model(feature)
        output = torch.squeeze(output)
        overall_accuracies += prediction_accuracy(output, torch.tensor(label).to(device))
    print('Epoch #%d Loss %.3f' % (epoch, curr_loss))
    recorded_losses.append(curr_loss)
    print(overall_accuracies)
    recorded_accuracies.append(overall_accuracies)

model.eval()    
masks = torch.zeros(7).to(device)
train_accuracies = torch.zeros(7).to(device)
for feature, label in zip(train_inputs, train_labels):
    feature = torch.tensor(feature, dtype=torch.float32).to(device)
    output = model(feature)
    output = torch.squeeze(output)
    train_accuracies += prediction_accuracy(output, torch.tensor(label).to(device))

print(train_accuracies)
# print("Final Accuracy: " + test_accuracies)
#torch.save(model,'model.pt')


masks = torch.zeros(7).to(device)
test_accuracies = torch.zeros(7).to(device)
for feature, label in zip(test_inputs, test_labels):
    feature = torch.tensor(feature, dtype=torch.float32).to(device)
    output = model(feature)
    output = torch.squeeze(output)
    masks += torch.tensor(label[0]).to(device)
    test_accuracies += prediction_accuracy(output, torch.tensor(label[1]).to(device), torch.tensor(label[0]).to(device))

print(test_accuracies)
test_accuracies = torch.div(test_accuracies, masks)
# print("Final Accuracy: " + test_accuracies)
torch.save(model,'LSTMmodel.pt')
print(test_accuracies)
results = {"Train Loss": recorded_losses, "Train Accuracies": recorded_accuracies, "Test Accuracy": test_accuracies}
results_dst = Path(f"./LSTMresults.pkl")
results_dst.write_bytes(pickle.dumps(results))