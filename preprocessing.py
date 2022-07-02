import pandas as pd

data = pd.read_csv('song_annotations.txt', delimiter='\t')
data.columns = ["song", "label"]
print(data)
print(data.label.unique())
# emotions = ["aggressive", "angry", "arousing", "boring", "calming", "cheerful", "cold", 
# "depressed", "emotional", "exciting", "happy", "lighthearted", "mellow", "morose", "negative", "negative feelings", "normal",
# "positive", "relax", "romantic", "sad", "strong", ""] #also have not-angry, not-happy, etc.. do we include??
# data_emotions = pd.loc()