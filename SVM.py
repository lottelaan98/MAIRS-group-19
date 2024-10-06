import numpy as np
import pandas as pd
from DifficultCases import DifficultCases
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SVM:
    """
    Support Vector Machine classifies utterances based on dialog acts.
    """
    def __init__(self, dataset):
        self.original_dataset = dataset
        self.dataset_without_duplicates = self.original_dataset.drop_duplicates(subset=['utterance content'])
        self.vectorizer = TfidfVectorizer()  # Adjust parameters as needed
        self.svm_classifier = None
        self.difficult_cases = DifficultCases()

    def find_x_and_y(self, dataset):
        x = self.vectorizer.fit_transform(dataset['utterance content'])
        y = dataset['dialog act']
        return x, y

    def train_and_test(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        self.svm_classifier = SVC()  # You can add parameters here for tuning
        self.svm_classifier.fit(x_train, y_train)
        y_pred = self.svm_classifier.predict(x_test)
        return y_test, y_pred

    def make_prediction(self, x_test):
        if self.svm_classifier is None:
            raise Exception("Model is not trained yet. Call train_and_test() first.")
        y_pred = self.svm_classifier.predict(x_test)
        return y_pred

    def report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)

    def print_wrong_predictions(self, y_test, y_pred, dialog_act=None):
        # Convert y_test to an array (if it's a pandas Series)
        y_test = y_test.values if hasattr(y_test, 'values') else y_test

        # Iterate through predictions and compare them with true labels
        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                if dialog_act is None or y_test[i] == dialog_act:
                    original_row = self.original_dataset.iloc[i]
                    print(f"Index {i}: Original utterance: {original_row['utterance content']}")
                    print(f"True label = {y_test[i]}, Predicted label = {y_pred[i]}")
                    print("-" * 80)

    def process_difficult_cases(self):
        x, y = self.find_x_and_y(self.original_dataset)
        self.train_and_test(x, y)
        self.difficult_cases.process_difficult_cases(self.svm_classifier, self.vectorizer)

