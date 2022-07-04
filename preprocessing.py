import pandas as pd


data = pd.read_csv('song_annotations.txt', delimiter='\t')
data.columns = ["song", "label"]
print(data)
print(data.label.unique())
emotions = ["aggressive", "angry", "arousing", "boring", "calming", "cheerful", "cold", 
"depressed", "emotional", "exciting", "happy", "lighthearted", "mellow", "morose", "negative", "negative feelings", "normal",
"not angry - agressive", "not calming", "not happy", "not sad", "pleasant", "positive", "relax", "romantic", "rough", "sad", 
"strong", "tender", "touching", "unpleasant", "unromantic"] 
data_emotions = data.loc[data['label'].isin(emotions)]
print(data_emotions)