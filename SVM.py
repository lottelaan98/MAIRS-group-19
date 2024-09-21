import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# DIT DEEL MAG IN GENERAL
class SVM:
    """
    Support Vector Machine classifies utterances based on the most common class
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def vectorize(self, dialog_acts, utterances):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(utterances)
        y = dialog_acts
        return X, y

    def split_data(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        return x_train, x_test, y_train, y_test

    def fit_svm(self, x_train, y_train):
        svm = SVC()
        svm.fit(x_train, y_train)
        return svm
        
    def make_prediction(self, x_test, svm):
        y_pred = svm.predict(x_test)
        return y_pred

    def report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred)
        return report

    def perform_svm(self):
        dialog_acts = self.dataset['dialog act']
        utterances = self.dataset['utterance content']
        x, y = self.vectorize(dialog_acts, utterances)
        x_train, x_test, y_train, y_test = self.split_data(x, y)
        fitted_svm = self.fit_svm(x_train, y_train)
        y_pred = self.make_prediction(x_test, fitted_svm)

        # Create a report of the results
        class_report = self.report(y_test, y_pred)
        print(class_report)
        
        