import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score


class Baseline1:
    """
    Baseline 1 classifies utterances based on the most common class.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_without_duplicates = self.dataset.drop_duplicates(
            subset=["utterance content"]
        )
        self.most_frequent_class = None
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.find_most_frequent_dialog_act()

    def find_most_frequent_dialog_act(self):
        word_counts = Counter(self.dataset["dialog act"])
        self.most_frequent_class = word_counts.most_common(1)[0][0]

    def classify(self):
        return self.most_frequent_class

    def find_x_and_y(self, dataset):
        x = self.vectorizer.fit_transform(dataset["utterance content"])
        y = dataset["dialog act"]
        return x, y

    def train_and_test(self, x, y):
        _, _, _, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        y_pred = [self.most_frequent_class] * len(
            y_test
        )  # All predictions are the most common class
        return y_test, y_pred
