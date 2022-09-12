import torch
from pathlib import Path
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
allLabels = torch.zeros(7).to(device)
allMasks = torch.zeros(7).to(device)
melInputs_dst = Path(f"./melInputs.pkl")
mfccInputs_dst = Path(f"./mfccInputs.pkl")
labels_dst = Path(f"./labels.pkl")
melinputs = pd.read_pickle(melInputs_dst)
mfccinputs = pd.read_pickle(mfccInputs_dst)
labels = pd.read_pickle(labels_dst)
print(input)
print(len(melinputs))
print(len(melinputs[0]))
print(len(melinputs[0][0]))
print(len(mfccinputs[0][0]))
# for features, label in zip(inputs, labels):
    # allLabels += torch.tensor(label[1]).to(device)
    # allMasks += torch.tensor(label[0]).to(device)
# print(allLabels)
# print(allMasks)
# print(allLabels / allMasks)


    