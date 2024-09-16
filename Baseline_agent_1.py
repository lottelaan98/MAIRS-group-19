import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Frequently used strings
dialogAct = 'dialog act'
utteranceContent = 'utterance content'

# Location of the the data file. CHANGE THIS ACCORDING TO THE PATH ON YOUR OWN COMPUTER
file_path = "dialog_acts.dat"
dataset = pd.read_csv(file_path)

# Load the data into a DataFrame
df = pd.read_csv(file_path, delimiter='\t', header=None)

# Split each row into 'dialog act' and 'utterance content'
df[dialogAct] = df[0].apply(lambda x: x.split(' ', 1)[0])
df[utteranceContent] = df[0].apply(lambda x: x.split(' ', 1)[1])
df = df.drop(columns=[0])

class baseline1:
    def __init__(self, dataset):
        self.data = dataset 
        self.most_frequent_word = None
        self.find_most_frequent_dialog_act()
    
    def find_most_frequent_dialog_act(self):
        dataset = self.data
        first_dialog_act = dataset.iloc[:, 0].apply(lambda sentence: sentence.split()[0] if isinstance(sentence, str) else None)
        word_counts = Counter(first_dialog_act)
        self.most_frequent_word = word_counts.most_common(1)[0][0]
    
    def classify(self, sentence):
        return self.most_frequent_word
    
classifier = baseline1(dataset)
print('majority class = ', classifier.most_frequent_word)
