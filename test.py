import torch
from pathlib import Path
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
allLabels = torch.zeros(32).to(device)
allMasks = torch.zeros(32).to(device)
inputs_dst = Path(f"./inputs.pkl")
labels_dst = Path(f"./labels.pkl")
inputs = pd.read_pickle(inputs_dst)
labels = pd.read_pickle(labels_dst)
for features, label in zip(inputs, labels):
    allLabels += torch.tensor(label[1]).to(device)
    allMasks += torch.tensor(label[0]).to(device)
print(allLabels)
print(allMasks)
print(allLabels / allMasks)


    