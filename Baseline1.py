import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

class Baseline1:
    """
    Baseline 1 classifies utterances based on the most common class
    """
    def __init__(self, dataset):
        self.dataset = dataset 
        self.most_frequent_class = None
        self.find_most_frequent_dialog_act()
    
    def find_most_frequent_dialog_act(self):
        word_counts = Counter(self.dataset.iloc[:, 0])
        self.most_frequent_class = word_counts.most_common(1)[0][0]
    
    def classify(self):
        return self.most_frequent_class
    