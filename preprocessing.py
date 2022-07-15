import pandas as pd
import numpy as np
import glob

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
for index, row in data_emotions.iterrows():
    try:
        new_dict[row['song']]
    except KeyError:
        new_dict[row['song']] = [row['label']]
    else:
        new_dict[row['song']].append(row['label'])
print(new_dict)
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
print(len(mel))
print(len(mfcc))
#use sigmoid for multilabel
