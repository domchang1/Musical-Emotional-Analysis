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
    output = (output >= 0.0).long()
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
melInputs_dst = Path(f"./melInputs.pkl")
mfccInputs_dst = Path(f"./mfccInputs.pkl")
labels_dst = Path(f"./labels.pkl")
melinputs = pd.read_pickle(melInputs_dst)
mfccinputs = pd.read_pickle(mfccInputs_dst)
labels = pd.read_pickle(labels_dst)
new_melinputs = []
new_mfccinputs = []
new_labels = []
for i in order:
    new_melinputs.append(melinputs[i])
    new_mfccinputs.append(mfccinputs[i])
    new_labels.append(labels[i])
melinputs = new_melinputs
mfccinputs = new_mfccinputs
labels = new_labels


inputs = 0
train_melinputs = melinputs[:351]
train_mfccinputs = mfccinputs[:351]
train_labels = labels[:351]
test_melinputs = melinputs[351:]
test_mfcciinputs = mfccinputs[351:]
test_labels = labels[351:]

# input = torch.tensor(inputs[0], dtype=torch.float32)

model = LSTM.LSTM(90, 7, 128).to(device)
criterion = nn.BCEWithLogitsLoss() #pos_weight=torch.ones([32])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
recorded_losses = []
recorded_accuracies = []
overall_accuracies = torch.zeros(7).to(device)
model.eval()
masks = torch.zeros(7).to(device)
curr_loss = 0
for mel, mfcc, label in zip(train_melinputs, train_mfccinputs, train_labels):
    #add channel dimension, check Conv2d
    mel = torch.tensor(mel, dtype=torch.float32).to(device)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).to(device)
    output = model(mel, mfcc)
    output = torch.squeeze(output)
    masks += torch.tensor(label[0]).to(device)
    loss = criterion(output, torch.tensor(label[1]).to(device))
    loss = torch.inner(loss, torch.tensor(label[0]).to(device))
    loss = torch.sum(loss)
    curr_loss += loss.item()
    overall_accuracies += prediction_accuracy(output, torch.tensor(label[1]).to(device), torch.tensor(label[0]).to(device))
print('Loss %.3f' % (curr_loss))

overall_accuracies /= masks
print(overall_accuracies)
    

for epoch in range(50):
    curr_loss = 0.0
    model.train()
    overall_accuracies = torch.zeros(7).to(device)
    for mel, mfcc, label in zip(train_melinputs, train_mfccinputs, train_labels):
        optimizer.zero_grad()
        mel = torch.tensor(mel, dtype=torch.float32).to(device)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).to(device)
        # print(data.shape)
        output = model(mel, mfcc)
        output = torch.squeeze(output)
        loss = criterion(output, torch.tensor(label[1]).to(device))
        loss = torch.inner(loss, torch.tensor(label[0]).to(device))
        loss = torch.sum(loss)
        curr_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    model.eval()
    masks = torch.zeros(7).to(device)
    for mel, mfcc, label in zip(train_melinputs, train_mfccinputs, train_labels):
        mel = torch.tensor(mel, dtype=torch.float32).to(device)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).to(device)
        output = model(mel, mfcc)
        output = torch.squeeze(output)
        masks += torch.tensor(label[0]).to(device)
        overall_accuracies += prediction_accuracy(output, torch.tensor(label[1]).to(device), torch.tensor(label[0]).to(device))
    print('Epoch #%d Loss %.3f' % (epoch, curr_loss))
    recorded_losses.append(curr_loss)
    overall_accuracies /= masks
    print(overall_accuracies)
    recorded_accuracies.append(overall_accuracies)

model.eval()    
masks = torch.zeros(7).to(device)
train_accuracies = torch.zeros(7).to(device)
for mel, mfcc, label in zip(train_melinputs, train_mfccinputs, train_labels):
    mel = torch.tensor(mel, dtype=torch.float32).to(device)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).to(device)
    output = model(mel, mfcc)
    output = torch.squeeze(output)
    masks += torch.tensor(label[0]).to(device)
    train_accuracies += prediction_accuracy(output, torch.tensor(label[1]).to(device), torch.tensor(label[0]).to(device))

print(train_accuracies)
print(masks)
train_accuracies = torch.div(train_accuracies, masks)
# print("Final Accuracy: " + test_accuracies)
#torch.save(model,'model.pt')
print(train_accuracies)


masks = torch.zeros(7).to(device)
test_accuracies = torch.zeros(7).to(device)
for mel, mfcc, label in zip(train_melinputs, train_mfccinputs, train_labels):
    mel = torch.tensor(mel, dtype=torch.float32).to(device)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).to(device)
    output = model(mel, mfcc)
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



