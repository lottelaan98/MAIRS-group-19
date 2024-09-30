import numpy as np
import pandas as pd
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

    def difficult_cases(self):
        keywords = {
            "ack": [
                "alright", "nice", "looking forward", "looking forward to it", "perfect",
                "excellent", "splendid", "very good", "couldn't be better", "can't be better",
                "great", "that's great", "wonderful", "fantastic", "awesome", "brilliant"
            ],
            "confirm": [
                "and the address", "and the location", "and the food type", "and the cuisine",
                "and the area", "and the pricerange", "and the reservation details", "and the date",
                "and the parking possibilities", "and the payment method"
            ],
            "repeat": [
                "sorry", "repeat that please", "could you repeat", "can you repeat", 
                "previous", "previous please", "try again", "pardon", "i missed that",
                "one more time", "say that again", "i didn't catch that", "rewind", 
                "go over that again", "what did you say", "please go over that again", 
                "can you clarify", "would you mind repeating"
            ]
        }

        data = [(key, phrase) for key, phrases in keywords.items() for phrase in phrases]
        df = pd.DataFrame(data, columns=['dialog act', 'utterance content'])
        return df

    def perform_svm_difficult(self):
        df_difficult = self.difficult_cases()
        x, y = self.find_x_and_y(self.original_dataset)
        x_vec_diff, y_vec_diff = self.find_x_and_y(df_difficult)
        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.15, random_state=42)

        # Fit the SVM on the training dataset
        self.svm_classifier = SVC()
        self.svm_classifier.fit(x_train, y_train)

        # Make predictions on the difficult cases
        y_pred = self.make_prediction(x_vec_diff)

        # Create a report of the results
        class_report = self.report(y_vec_diff, y_pred)
        print(class_report)
