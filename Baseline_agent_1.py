import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

class baseline:
    def __init__(self, dataset):
        self.data = dataset 
        self.most_frequent_word = None
        self._train()
    
    def _train(self):
        dataset = self.data
        first_word = dataset.iloc[:, 0].apply(lambda sentence: sentence.split()[0] if isinstance(sentence, str) else None)
        word_counts = Counter(first_word)
        self.most_frequent_word = word_counts.most_common(1)[0][0]
    
    def classify(self, sentence):
        return self.most_frequent_word
    

file_path = "/Users/youssefbenmansour/Downloads/dialog_acts.dat"
dataset = pd.read_csv(file_path)

X_train, X_test = train_test_split(dataset, test_size=0.15, random_state=42)
classifier = baseline_agent_1(X_train)
