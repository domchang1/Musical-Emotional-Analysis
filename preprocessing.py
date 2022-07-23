import pandas as pd
import numpy as np
import glob
import pickle
from pathlib import Path

def processData():
    data = pd.read_csv('song_annotations.txt', delimiter='\t')
    data.columns = ["song", "label"]
    print(data)
    print(data.label.unique())
    emotions = ["aggressive", "angry", "arousing", "boring", "calming", "cheerful", "cold", 
    "depressed", "emotional", "exciting", "happy", "lighthearted", "mellow", "morose", "negative", "negative feelings", "normal",
    "not angry - agressive", "not calming", "not happy", "not sad", "pleasant", "positive", "relax", "romantic", "rough", "sad", 
    "strong", "tender", "touching", "unpleasant", "unromantic"] 
    data_emotions = data.loc[data['label'].isin(emotions)]
    data_emotions.sort_values(by='song', axis=0, inplace=True)
    print(data_emotions)
    new_dict = {}
    for _, row in data_emotions.iterrows():
        try:
            new_dict[row['song']]
        except KeyError:
            new_dict[row['song']] = [row['label']]
        else:
            new_dict[row['song']].append(row['label'])
    print(new_dict)
    emotions.remove('negative feelings')
    emotions.remove('not angry - agressive')
    emotions.remove('not calming')
    emotions.remove('not happy')
    emotions.remove('not sad')
    emotions.remove('unpleasant')
    emotions.remove('unromantic')
    emotions.append('not angry')
    emotions.append('not aggressive')
    emotions.sort()
    

    for key in new_dict:
        labels = new_dict[key]
        if 'negative feelings' in labels and 'negative' in labels:
            labels.remove('negative feelings')
        elif 'negative feelings' in labels:
            labels.remove('negative feelings')
            labels.append('negative')
        if 'not angry - agressive' in labels:
            labels.remove('not angry - agressive')
            labels.append('not angry')
            labels.append('not aggressive')
        labels.sort()
        mask = np.zeroes_like(emotions)
        true_labels = np.zeroes_like(emotions)
        for i in range(len(mask)):
            if emotions[i] in labels:
                mask[i] = 1
            
            
        new_dict[key] = [mask]
    

    exit()
    filenames = glob.glob('./mel_features/*.mel')
    mel = {}
    mfcc = {}
    for filename in filenames:
        song = filename[15:len(filename)-4]
        mel[song] = np.loadtxt(filename, delimiter=',')
    filenames = glob.glob("./mfcc_features/*.mfcc")
    for filename in filenames:
        song = filename[16:len(filename)-5]
        mfcc[song] = np.loadtxt(filename, delimiter=',')
    features = {}
    for key in mel:
        features[key] = np.concatenate((mel[key], mfcc[key]), axis=1)
    labels = []
    inputs = []
    for key in new_dict:
        labels.append(new_dict[key])
        inputs.append(features[key])
    labels_dst = Path(f"./labels.pkl")
    inputs_dst = Path(f"./inputs.pkl")
    labels_dst.write_bytes(pickle.dumps(labels))
    inputs_dst.write_bytes(pickle.dumps(inputs))





